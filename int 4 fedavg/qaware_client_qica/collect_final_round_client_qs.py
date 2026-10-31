#!/usr/bin/env python3
"""Collect final-round client q/s artifacts under int 4 fedavg.

This wrapper delegates the proven final-round collector while changing the
default output root to ``int 4 fedavg`` and handling fresh-start deletion safely.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_COLLECTOR = REPO_ROOT / "int4_client_qs_randomized_ica" / "collect_client_qs_final_round.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final-round client int4 q/s for QICA.")
    parser.add_argument("--ds", type=str, default="imagenet", choices=["imagenet"])
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg16"])
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--samples_per_client", type=int, default=8)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--private_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--store_scope", type=str, default="all", choices=["all", "attack_layer"])
    parser.add_argument("--fresh_start", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    root = allowed_root.resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to delete outside {root}: {target}")
    if target.exists():
        shutil.rmtree(target)


def main() -> None:
    args = parse_args()
    if not LEGACY_COLLECTOR.is_file():
        raise FileNotFoundError(LEGACY_COLLECTOR)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            REPO_ROOT
            / "int 4 fedavg"
            / args.ds
            / args.model
            / f"clients{args.num_clients}_rounds{args.rounds}_spc{args.samples_per_client}_seed{args.seed}"
        )
    else:
        output_dir = resolve_path(output_dir)
    if args.fresh_start:
        safe_rmtree(output_dir, allowed_root=REPO_ROOT / "int 4 fedavg")

    cmd = [
        sys.executable,
        str(LEGACY_COLLECTOR),
        "--ds",
        args.ds,
        "--model",
        args.model,
        "--num_clients",
        str(args.num_clients),
        "--rounds",
        str(args.rounds),
        "--samples_per_client",
        str(args.samples_per_client),
        "--local_epochs",
        str(args.local_epochs),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--h_dim",
        str(args.h_dim),
        "--num_workers",
        str(args.num_workers),
        "--private_split",
        args.private_split,
        "--output_dir",
        str(output_dir),
        "--store_scope",
        args.store_scope,
    ]
    if args.local_batch_size is not None:
        cmd.extend(["--local_batch_size", str(args.local_batch_size)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
