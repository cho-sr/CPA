from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import apply_overrides, load_config
from .quantizer import assert_cell_membership, quantize
from .run import append_jsonl, git_sha
from .seed import seed_all, torch_generator
from .uia import (
    load_or_collect_uia_feature_batch,
    load_vgg16_classifier,
    mp_prediction_from_quantized,
    numerical_rank_from_singular_values,
    resolve_uia_device,
    singular_values_from_factors,
)
from .victim import load_imagenet_dataset, select_label_indices


@dataclass
class TrainingFeatureBatch:
    labels: torch.Tensor
    image_paths: list[str]
    Z: torch.Tensor
    seed: int
    n: int
    batch_index: int
    data_root: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedSGD round-dependence diagnostic: FC-only checkpoints, spectrum only."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-tag", default="m6_round_dependence")
    parser.add_argument("--output-dir", type=Path, default=Path("results/m6_round_dependence"))
    parser.add_argument("--rounds", default="0,5,20,50,100")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--batch-indices", default="0,1,2,3,4")
    parser.add_argument("--measurement-split", default="val")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-batch-indices", default="0,1,2,3,4")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--train-cache-dir", type=Path, default=Path("results/m6_round_dependence/train_feature_cache"))
    parser.add_argument("--force-train-cache", action="store_true")
    parser.add_argument("--uia-dump-steps", type=int, default=10)
    parser.add_argument("--uia-dump-dir", type=Path, default=Path("results/m6_round_dependence/uia_T10_dumps"))
    parser.add_argument("--skip-uia-dump", action="store_true")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--rounding", default="stochastic")
    parser.add_argument("--scale-granularity", default="per_row")
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--c-mp", type=float, default=3.0)
    parser.add_argument("--rank-rel-tol", type=float, default=1.0e-5)
    parser.add_argument("--rank-abs-tol", type=float, default=1.0e-8)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = apply_overrides(load_config(args.config), args.overrides)
    cfg = condition_config(base_config, args)
    seed_all(int(cfg["seed"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_uia_device(cfg)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    rounds = parse_ints(args.rounds)
    if not rounds or min(rounds) < 0:
        raise ValueError(f"rounds must be non-negative, got {args.rounds!r}")
    max_round = max(rounds)
    batch_indices = parse_ints(args.batch_indices)
    if not batch_indices:
        raise ValueError("batch-indices must not be empty")
    train_batch_indices = parse_ints(args.train_batch_indices)
    if not train_batch_indices:
        raise ValueError("train-batch-indices must not be empty")

    print(
        "round dependence device={device} name={name} n={n} measurement={msplit}:{batches} "
        "train={tsplit}:{train_batches} rounds={rounds}".format(
            device=device,
            name=device_name,
            n=int(args.n),
            msplit=str(args.measurement_split),
            batches=batch_indices,
            tsplit=str(args.train_split),
            train_batches=train_batch_indices,
            rounds=rounds,
        ),
        flush=True,
    )

    measure_cfg = deepcopy(cfg)
    measure_cfg.setdefault("data", {})["split"] = str(args.measurement_split)
    features = [
        load_or_collect_uia_feature_batch(
            measure_cfg,
            n=int(args.n),
            batch_index=int(batch_index),
            force=bool(args.force_cache),
        )
        for batch_index in batch_indices
    ]
    train_features = [
        load_or_collect_training_feature_batch(
            cfg,
            split=str(args.train_split),
            n=int(args.train_batch_size),
            batch_index=int(batch_index),
            cache_dir=Path(args.train_cache_dir),
            device=device,
            force=bool(args.force_train_cache),
        )
        for batch_index in train_batch_indices
    ]
    classifier = load_vgg16_classifier(device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=float(args.lr), momentum=0.0, weight_decay=0.0)

    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    uia_dump_rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()

    for current_round in range(max_round + 1):
        if current_round in rounds:
            checkpoint_start = time.perf_counter()
            training_loss = mean_probe_loss(classifier, train_features, device=device)
            measurement_probe_loss = mean_probe_loss(classifier, features, device=device)
            training_rows.append(
                {
                    "run_tag": str(args.run_tag),
                    "round": int(current_round),
                    "training_loss_mean": float(training_loss),
                    "measurement_probe_loss_mean": float(measurement_probe_loss),
                    "training_schedule": "cycled_train_split_batches_fc_only",
                    "train_split": str(args.train_split),
                    "measurement_split": str(args.measurement_split),
                    "lr": float(args.lr),
                    "train_batch_size": int(args.train_batch_size),
                    "n_measurement": int(args.n),
                    "n_train_batches": len(train_features),
                    "n_measurement_batches": len(features),
                }
            )
            for feature in features:
                measured = measure_batch(
                    args=args,
                    feature=feature,
                    classifier=classifier,
                    device=device,
                    device_name=device_name,
                    round_index=int(current_round),
                    training_loss=float(training_loss),
                    measurement_probe_loss=float(measurement_probe_loss),
                    checkpoint_start=checkpoint_start,
                )
                summary_rows.append(measured["summary"])
                source_rows.extend(measured["source_rows"])
                spectrum_rows.extend(measured["spectrum_rows"])
                if not bool(args.skip_uia_dump) and int(args.uia_dump_steps) > 0:
                    uia_dump_rows.append(
                        dump_uia_delta(
                            args=args,
                            feature=feature,
                            classifier=classifier,
                            device=device,
                            round_index=int(current_round),
                            output_dir=Path(args.uia_dump_dir),
                        )
                    )
            write_outputs(output_dir, summary_rows, source_rows, spectrum_rows, training_rows, uia_dump_rows)
            print_round_status(summary_rows, round_index=int(current_round))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if current_round < max_round:
            train_feature = train_features[current_round % len(train_features)]
            train_one_round(classifier, optimizer, train_feature, device=device)

    result = {
        "schema_version": 1,
        "run_tag": str(args.run_tag),
        "milestone": "m6_round_dependence",
        "git_sha": git_sha(),
        "protocol": {
            "fl_alg": "fedsgd",
            "model_state": "fc_only_checkpoint",
            "training_schedule": "cycled_train_split_batches_fc_only",
            "train_split": str(args.train_split),
            "measurement_split": str(args.measurement_split),
            "measurement_batches_are_used_for_training": False,
            "local_steps_observed": 1,
            "reconstruction": "disabled",
        },
        "config": {
            **deepcopy(cfg),
            "round_dependence": {
                "rounds": rounds,
                "n": int(args.n),
                "batch_indices": batch_indices,
                "measurement_split": str(args.measurement_split),
                "train_split": str(args.train_split),
                "train_batch_indices": train_batch_indices,
                "train_batch_size": int(args.train_batch_size),
                "bits": int(args.bits),
                "rounding": str(args.rounding),
                "scale_granularity": str(args.scale_granularity),
                "bucket_size": int(args.bucket_size),
                "lr": float(args.lr),
                "c_mp": float(args.c_mp),
                "uia_dump_steps": int(args.uia_dump_steps),
                "uia_dump_enabled": not bool(args.skip_uia_dump),
                "output_dir": str(output_dir),
            },
        },
        "metrics": {
            "summary": summary_rows,
            "training_loss": training_rows,
            "uia_dumps": uia_dump_rows,
        },
        "artifacts": {
            "summary_csv": str(output_dir / "summary.csv"),
            "source_norms_csv": str(output_dir / "source_norms.csv"),
            "lam_spectrum_csv": str(output_dir / "lam_spectrum.csv"),
            "training_loss_csv": str(output_dir / "training_loss.csv"),
            "uia_dump_index_csv": str(output_dir / "uia_T10_dump_index.csv"),
            "summary_png": str(output_dir / "round_dependence_summary.png"),
        },
        "wall_time_sec": float(time.perf_counter() - start_all),
    }
    append_jsonl(Path(cfg["paths"]["results_jsonl"]), result)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_summary(summary_rows, training_rows, output_dir / "round_dependence_summary.png")
    print(f"round dependence complete in {result['wall_time_sec']:.1f}s", flush=True)


def condition_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg["run_tag"] = str(args.run_tag)
    cfg.setdefault("data", {})["batch_size"] = int(args.n)
    cfg["data"]["split"] = str(args.measurement_split)
    cfg["data"]["n_batches"] = 1
    cfg["data"]["microbatch_size"] = min(32, int(args.n))
    cfg["data"]["require_distinct_labels"] = True
    cfg.setdefault("stage1", {})["n_components"] = int(args.n)
    cfg["stage1"]["c_mp"] = float(args.c_mp)
    cfg.setdefault("quantization", {})["bits"] = int(args.bits)
    cfg["quantization"]["rounding"] = str(args.rounding)
    cfg["quantization"]["scale_granularity"] = str(args.scale_granularity)
    cfg["quantization"]["bucket_size"] = int(args.bucket_size)
    return cfg


def load_or_collect_training_feature_batch(
    config: dict[str, Any],
    *,
    split: str,
    n: int,
    batch_index: int,
    cache_dir: Path,
    device: torch.device,
    force: bool = False,
) -> TrainingFeatureBatch:
    seed = int(config["seed"])
    path = (
        Path(cache_dir)
        / f"training_features_seed{seed}_split{split}_n{int(n)}_batch{int(batch_index)}.pt"
    )
    if path.is_file() and not force:
        data = torch.load(path, map_location="cpu")
        return TrainingFeatureBatch(
            labels=data["labels"].long(),
            image_paths=list(data["image_paths"]),
            Z=data["Z"].float(),
            seed=int(data["seed"]),
            n=int(data["n"]),
            batch_index=int(data["batch_index"]),
            data_root=str(data["data_root"]),
            split=str(data["split"]),
        )

    dataset = load_imagenet_dataset(Path(config["data"]["root"]), str(split))
    indices = select_label_indices(
        dataset,
        B=int(n),
        seed=seed,
        batch_index=int(batch_index),
        require_distinct_labels=True,
    )
    samples = [dataset[int(idx)] for idx in indices]
    x = torch.stack([sample[0] for sample in samples], dim=0)
    labels = torch.tensor([int(sample[1]) for sample in samples], dtype=torch.long)
    image_paths = [sample_path(dataset, int(idx)) for idx in indices]
    Z = extract_vgg16_features(x, device=device)
    out = TrainingFeatureBatch(
        labels=labels.cpu(),
        image_paths=image_paths,
        Z=Z.cpu().float().contiguous(),
        seed=seed,
        n=int(n),
        batch_index=int(batch_index),
        data_root=str(config["data"]["root"]),
        split=str(split),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "labels": out.labels,
            "image_paths": list(out.image_paths),
            "Z": out.Z,
            "seed": out.seed,
            "n": out.n,
            "batch_index": out.batch_index,
            "data_root": out.data_root,
            "split": out.split,
        },
        path,
    )
    return out


def extract_vgg16_features(x: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    from torchvision.models import VGG16_Weights, vgg16

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    chunks = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), 32):
            stop = min(start + 32, int(x.shape[0]))
            x_dev = x[start:stop].to(device=device, dtype=torch.float32)
            features = model.features(x_dev)
            pooled = model.avgpool(features)
            chunks.append(torch.flatten(pooled, 1).detach().cpu().float())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0).contiguous()


def sample_path(dataset: Any, idx: int) -> str:
    samples = getattr(dataset, "samples", None) or getattr(dataset, "imgs", None)
    if samples is None:
        return ""
    return str(samples[int(idx)][0])


def train_one_round(
    classifier: torch.nn.Sequential,
    optimizer: torch.optim.Optimizer,
    feature: Any,
    *,
    device: torch.device,
) -> float:
    Z = feature.Z.to(device=device, dtype=torch.float32)
    labels = feature.labels.to(device=device)
    optimizer.zero_grad(set_to_none=True)
    logits = classifier(Z)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def mean_probe_loss(classifier: torch.nn.Sequential, features: list[Any], *, device: torch.device) -> float:
    values = []
    with torch.no_grad():
        for feature in features:
            Z = feature.Z.to(device=device, dtype=torch.float32)
            labels = feature.labels.to(device=device)
            logits = classifier(Z)
            values.append(F.cross_entropy(logits, labels, reduction="mean").detach().cpu())
    return float(torch.stack(values).mean().item())


def measure_batch(
    *,
    args: argparse.Namespace,
    feature: Any,
    classifier: torch.nn.Sequential,
    device: torch.device,
    device_name: str,
    round_index: int,
    training_loss: float,
    measurement_probe_loss: float,
    checkpoint_start: float,
) -> dict[str, Any]:
    A, Y, logits, per_sample_loss = fedsgd_attack_layer_gradient(classifier, feature, device=device)
    rank_sv = singular_values_from_factors(A, feature.Z, device=device)
    rank_svd, rank_tol = numerical_rank_from_singular_values(
        rank_sv,
        rel_tol=float(args.rank_rel_tol),
        abs_tol=float(args.rank_abs_tol),
    )
    quant_seed = quantization_seed(
        seed=int(feature.seed),
        batch_index=int(feature.batch_index),
        n=int(args.n),
        round_index=int(round_index),
    )
    qobs = quantize(
        Y,
        bits=int(args.bits),
        rounding=str(args.rounding),
        generator=torch_generator(quant_seed, device=Y.device),
        scale_granularity=str(args.scale_granularity),
        bucket_size=int(args.bucket_size),
    )
    assert_cell_membership(Y, qobs.Y_hat, qobs.delta)
    mp = mp_prediction_from_quantized(
        qobs.Y_hat,
        qobs.delta,
        requested_rank=int(args.n),
        rounding=qobs.rounding,
        c_mp=float(args.c_mp),
        device=device,
    )
    a_norm = A.norm(dim=0)
    stats = vector_stats(a_norm)
    spearman_loss_a = spearmanr(per_sample_loss.numpy(), a_norm.numpy())
    round0_a_rel = rel_l2(A, feature.gia_A) if int(round_index) == 0 else float("nan")
    round0_y_rel = rel_l2(Y, feature.gia_Y) if int(round_index) == 0 else float("nan")
    batch_loss = float(per_sample_loss.mean().item())
    summary = {
        "run_tag": str(args.run_tag),
        "round": int(round_index),
        "seed": int(feature.seed),
        "batch_index": int(feature.batch_index),
        "n": int(args.n),
        "bits": int(args.bits),
        "rounding": str(args.rounding),
        "scale_granularity": str(args.scale_granularity),
        "bucket_size": int(args.bucket_size) if str(args.scale_granularity) == "per_bucket" else "",
        "lr": float(args.lr),
        "c_mp": float(args.c_mp),
        "device": str(device),
        "device_name": str(device_name),
        "training_schedule": "cycled_train_split_batches_fc_only",
        "train_split": str(args.train_split),
        "measurement_split": str(args.measurement_split),
        "measurement_batches_are_used_for_training": False,
        "train_batch_size": int(args.train_batch_size),
        "train_batch_indices": str(args.train_batch_indices),
        "training_loss_mean": float(training_loss),
        "measurement_probe_loss_mean": float(measurement_probe_loss),
        "batch_loss_mean": batch_loss,
        "rank_svd": int(rank_svd),
        "rank_tol": float(rank_tol),
        "B_eff_pred": int(mp["B_eff_pred"]),
        "sigma2": float(mp["sigma2"]),
        "mp_edge": float(mp["mp_edge"]),
        "mp_threshold": float(mp["mp_threshold"]),
        "lambda_max": float(mp["lam_top"][0]) if mp["lam_top"] else float("nan"),
        "lam_deb": json.dumps(mp["lam_top"]),
        "lam_over_mp_edge": json.dumps(mp["lam_over_mp_edge"]),
        "min_lam_over_mp_edge": float(mp["min_lam_over_mp_edge"]),
        "kept_min_lam_over_mp_edge": float(mp["kept_min_lam_over_mp_edge"]),
        "discarded_max_lam_over_mp_edge": float(mp["discarded_max_lam_over_mp_edge"]),
        "a_norm_mean": stats["mean"],
        "a_norm_std": stats["std"],
        "a_norm_cv": stats["cv"],
        "a_norm_min": stats["min"],
        "a_norm_max": stats["max"],
        "a_norm_min_over_max": stats["min_over_max"],
        "a_norm_max_over_min": stats["max_over_min"],
        "spearman_loss_a_norm": float(spearman_loss_a),
        "round0_gia_A_rel_l2": float(round0_a_rel),
        "round0_gia_Y_rel_l2": float(round0_y_rel),
        "quantization_seed": int(quant_seed),
        "delta_mean": float(qobs.delta.detach().cpu().float().mean().item()),
        "delta_min": float(qobs.delta.detach().cpu().float().min().item()),
        "delta_max": float(qobs.delta.detach().cpu().float().max().item()),
        "Y_fro_norm": float(Y.norm().item()),
        "Y_max_abs": float(Y.abs().max().item()),
        "wall_time_sec": float(time.perf_counter() - checkpoint_start),
    }
    return {
        "summary": summary,
        "source_rows": make_source_rows(args, feature, a_norm, per_sample_loss, round_index),
        "spectrum_rows": make_spectrum_rows(args, feature, mp, round_index),
    }


def fedsgd_attack_layer_gradient(
    classifier: torch.nn.Sequential,
    feature: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Z = feature.Z.to(device=device, dtype=torch.float32)
    labels = feature.labels.to(device=device)
    classifier.zero_grad(set_to_none=True)
    attacked = classifier[0]
    hidden = attacked(Z)
    hidden.retain_grad()
    logits = classifier[1:](hidden)
    per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
    loss = per_sample_loss.mean()
    loss.backward()
    if hidden.grad is None:
        raise RuntimeError("missing attacked-layer hidden gradients")
    A = hidden.grad.detach().T.cpu().float().contiguous()
    Y = A.matmul(feature.Z.cpu().float()).contiguous()
    return A, Y, logits.detach().cpu().float().contiguous(), per_sample_loss.detach().cpu().float().contiguous()


def dump_uia_delta(
    *,
    args: argparse.Namespace,
    feature: Any,
    classifier: torch.nn.Sequential,
    device: torch.device,
    round_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"round{int(round_index):03d}_batch{int(feature.batch_index):03d}_T{int(args.uia_dump_steps)}.pt"
    if path.is_file():
        return {
            "run_tag": str(args.run_tag),
            "round": int(round_index),
            "batch_index": int(feature.batch_index),
            "n": int(args.n),
            "local_steps": int(args.uia_dump_steps),
            "lr": float(args.lr),
            "measurement_split": str(args.measurement_split),
            "path": str(path),
            "status": "exists",
            "update_factor_rel_l2": "",
            "initial_loss_mean": "",
            "final_loss_mean": "",
        }

    local_classifier = deepcopy(classifier).to(device)
    local_classifier.eval()
    for parameter in local_classifier.parameters():
        parameter.requires_grad_(True)
    attacked = local_classifier[0]
    W0 = attacked.weight.detach().clone()
    optimizer = torch.optim.SGD(local_classifier.parameters(), lr=float(args.lr), momentum=0.0, weight_decay=0.0)
    Z = feature.Z.to(device=device, dtype=torch.float32)
    labels = feature.labels.to(device=device)
    B_cumulative = torch.zeros((attacked.out_features, Z.shape[0]), device=device, dtype=torch.float32)
    initial_loss = float("nan")
    final_loss = float("nan")
    with torch.enable_grad():
        for step in range(1, int(args.uia_dump_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            hidden = attacked(Z)
            hidden.retain_grad()
            logits = local_classifier[1:](hidden)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            if step == 1:
                initial_loss = float(per_sample_loss.mean().detach().cpu())
            loss = per_sample_loss.mean()
            loss.backward()
            if hidden.grad is None:
                raise RuntimeError("missing UIA hidden gradients")
            B_cumulative += hidden.grad.detach().T
            optimizer.step()
    with torch.no_grad():
        final_logits = local_classifier(Z)
        final_per_sample_loss = F.cross_entropy(final_logits, labels, reduction="none")
        final_loss = float(final_per_sample_loss.mean().detach().cpu())
    delta_over_lr = (W0 - attacked.weight.detach()) / float(args.lr)
    Y_factor = B_cumulative.matmul(Z.detach())
    update_rel = rel_l2(delta_over_lr.detach().cpu(), Y_factor.detach().cpu())
    torch.save(
        {
            "schema_version": 1,
            "run_tag": str(args.run_tag),
            "protocol": {
                "fl_alg": "fedavg",
                "conv": "frozen",
                "classifier": "fc_only",
                "local_steps": int(args.uia_dump_steps),
                "source_checkpoint": f"fedsgd_round_{int(round_index)}",
            },
            "round": int(round_index),
            "seed": int(feature.seed),
            "batch_index": int(feature.batch_index),
            "n": int(args.n),
            "measurement_split": str(args.measurement_split),
            "lr": float(args.lr),
            "labels": feature.labels.cpu().long(),
            "image_paths": list(feature.image_paths),
            "delta_theta_over_lr": delta_over_lr.detach().cpu().float().contiguous(),
            "B_cumulative": B_cumulative.detach().cpu().float().contiguous(),
            "initial_loss_mean": initial_loss,
            "final_loss_mean": final_loss,
            "update_factor_rel_l2": float(update_rel),
        },
        path,
    )
    del local_classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "run_tag": str(args.run_tag),
        "round": int(round_index),
        "batch_index": int(feature.batch_index),
        "n": int(args.n),
        "local_steps": int(args.uia_dump_steps),
        "lr": float(args.lr),
        "measurement_split": str(args.measurement_split),
        "path": str(path),
        "status": "written",
        "update_factor_rel_l2": float(update_rel),
        "initial_loss_mean": initial_loss,
        "final_loss_mean": final_loss,
    }


def make_source_rows(
    args: argparse.Namespace,
    feature: Any,
    a_norm: torch.Tensor,
    per_sample_loss: torch.Tensor,
    round_index: int,
) -> list[dict[str, Any]]:
    rows = []
    labels = [int(value) for value in feature.labels.tolist()]
    for idx in range(int(args.n)):
        rows.append(
            {
                "run_tag": str(args.run_tag),
                "round": int(round_index),
                "seed": int(feature.seed),
                "batch_index": int(feature.batch_index),
                "n": int(args.n),
                "source_index": int(idx),
                "label": labels[idx],
                "image_path": feature.image_paths[idx] if idx < len(feature.image_paths) else "",
                "a_norm": float(a_norm[idx].item()),
                "loss": float(per_sample_loss[idx].item()),
            }
        )
    return rows


def make_spectrum_rows(args: argparse.Namespace, feature: Any, mp: dict[str, Any], round_index: int) -> list[dict[str, Any]]:
    rows = []
    ratios = list(mp["lam_over_mp_edge"])
    lam_top = list(mp["lam_top"])
    B_eff = int(mp["B_eff_pred"])
    for idx, ratio in enumerate(ratios):
        rows.append(
            {
                "run_tag": str(args.run_tag),
                "round": int(round_index),
                "seed": int(feature.seed),
                "batch_index": int(feature.batch_index),
                "n": int(args.n),
                "component_rank": int(idx),
                "lam_deb": float(lam_top[idx]),
                "lam_over_mp_edge": float(ratio),
                "kept_by_c_mp": bool(idx < B_eff),
                "c_mp": float(args.c_mp),
                "mp_edge": float(mp["mp_edge"]),
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    uia_dump_rows: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "source_norms.csv", source_rows)
    write_csv(output_dir / "lam_spectrum.csv", spectrum_rows)
    write_csv(output_dir / "training_loss.csv", training_rows)
    write_csv(output_dir / "uia_T10_dump_index.csv", uia_dump_rows)


def render_summary(summary_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], output: Path) -> None:
    if not summary_rows:
        return
    rounds = sorted({int(row["round"]) for row in summary_rows})
    x = np.arange(len(rounds), dtype=np.float64)
    labels = [str(value) for value in rounds]
    grouped = {round_value: [row for row in summary_rows if int(row["round"]) == round_value] for round_value in rounds}

    def mean_std(key: str) -> tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for round_value in rounds:
            vals = np.asarray([float(row[key]) for row in grouped[round_value]], dtype=np.float64)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
        return np.asarray(means), np.asarray(stds)

    cv_mean, cv_std = mean_std("a_norm_cv")
    beff_mean, beff_std = mean_std("B_eff_pred")
    sp_mean, sp_std = mean_std("spearman_loss_a_norm")
    minmax_mean, minmax_std = mean_std("a_norm_min_over_max")
    train_by_round = {int(row["round"]): float(row["training_loss_mean"]) for row in training_rows}
    train_loss = np.asarray([train_by_round.get(round_value, np.nan) for round_value in rounds], dtype=np.float64)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("M6 Round Dependence: FedSGD Gradient Spectrum at FC-only Checkpoints", fontsize=16, fontweight="bold")
    ax_cv, ax_beff, ax_sp, ax_loss = axes.ravel()
    plot_error(ax_cv, x, cv_mean, cv_std, labels, "||a_i|| CV", "CV")
    ax_min = ax_cv.twinx()
    ax_min.plot(x, minmax_mean, color="#059669", marker="s", linewidth=1.5, label="min/max")
    ax_min.fill_between(x, minmax_mean - minmax_std, minmax_mean + minmax_std, color="#059669", alpha=0.10)
    ax_min.set_ylabel("min/max")
    ax_min.legend(loc="upper right", frameon=False)

    plot_error(ax_beff, x, beff_mean, beff_std, labels, "Predicted B_eff", "B_eff")
    ax_beff.set_ylim(0, max(32, float(np.nanmax(beff_mean + beff_std)) + 2))
    plot_error(ax_sp, x, sp_mean, sp_std, labels, "Spearman(loss, ||a_i||)", "rho")
    ax_sp.axhline(0.0, color="#111827", linewidth=1.0)
    ax_loss.plot(x, train_loss, color="#7c3aed", marker="o", linewidth=2.0)
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(labels)
    ax_loss.set_xlabel("round checkpoint")
    ax_loss.set_ylabel("mean training loss")
    ax_loss.set_title("FC-only training progress")
    ax_loss.grid(True, alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_error(
    ax: Any,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    labels: list[str],
    title: str,
    ylabel: str,
) -> None:
    ax.plot(x, mean, color="#2563eb", marker="o", linewidth=2.0)
    ax.fill_between(x, mean - std, mean + std, color="#2563eb", alpha=0.14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("round checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def print_round_status(rows: list[dict[str, Any]], *, round_index: int) -> None:
    subset = [row for row in rows if int(row["round"]) == int(round_index)]
    if not subset:
        return
    cv = np.asarray([float(row["a_norm_cv"]) for row in subset], dtype=np.float64)
    beff = np.asarray([float(row["B_eff_pred"]) for row in subset], dtype=np.float64)
    sp = np.asarray([float(row["spearman_loss_a_norm"]) for row in subset], dtype=np.float64)
    loss = np.asarray([float(row["batch_loss_mean"]) for row in subset], dtype=np.float64)
    print(
        "round={r} loss={loss:.4g} CV={cv:.4g} B_eff={beff:.3g} Spearman={sp:.4g}".format(
            r=int(round_index),
            loss=float(np.mean(loss)),
            cv=float(np.mean(cv)),
            beff=float(np.mean(beff)),
            sp=float(np.mean(sp)),
        ),
        flush=True,
    )


def vector_stats(values: torch.Tensor) -> dict[str, float]:
    vals = values.detach().cpu().float()
    mean = float(vals.mean().item())
    std = float(vals.std(unbiased=False).item())
    min_value = float(vals.min().item())
    max_value = float(vals.max().item())
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean != 0.0 else float("nan"),
        "min": min_value,
        "max": max_value,
        "min_over_max": float(min_value / max(max_value, 1.0e-30)),
        "max_over_min": float(max_value / max(min_value, 1.0e-30)),
    }


def rel_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    diff = left.detach().cpu().float() - right.detach().cpu().float()
    denom = right.detach().cpu().float().norm().clamp_min(torch.tensor(1.0e-12))
    return float((diff.norm() / denom).item())


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def quantization_seed(*, seed: int, batch_index: int, n: int, round_index: int) -> int:
    return int(seed) + 1_000_003 * int(batch_index) + 9_173 * int(n) + 19_999 * int(round_index)


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
