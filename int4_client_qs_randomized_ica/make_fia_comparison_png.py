#!/usr/bin/env python3
"""Create side-by-side FIA comparison PNGs from completed recon pickles."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "int4_client_qs_randomized_ica"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make original | naive dequant | randomized FIA comparison PNG."
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--R", type=int, default=64)
    parser.add_argument("--smoke_R", type=int, default=None)
    parser.add_argument("--n_sample", type=int, default=2)
    parser.add_argument("--n_iter_fi", type=int, default=25000)
    parser.add_argument("--naive_dir", type=Path, default=None)
    parser.add_argument("--randomized_dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--order", choices=["eval", "attack"], default="eval")
    parser.add_argument("--max_items", type=int, default=24)
    parser.add_argument("--item_cols", type=int, default=2)
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip missing branches instead of failing.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def default_randomized_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    effective_R = args.smoke_R if args.smoke_R is not None else args.R
    suffix = "_smoke" if args.smoke_R is not None else ""
    return run_dir / f"randomized_R{effective_R}{suffix}" / (
        f"direct_fia_n{args.n_sample}_iter{args.n_iter_fi}"
    )


def default_naive_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    return run_dir / "naive" / f"direct_fia_n{args.n_sample}_iter{args.n_iter_fi}"


def default_output_path(args: argparse.Namespace, run_dir: Path) -> Path:
    effective_R = args.smoke_R if args.smoke_R is not None else args.R
    suffix = "_smoke" if args.smoke_R is not None else ""
    return (
        run_dir
        / "comparison_png"
        / (
            f"original_naive_dequant_randomized_R{effective_R}{suffix}_"
            f"n{args.n_sample}_iter{args.n_iter_fi}_{args.order}_order.png"
        )
    )


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def to_uint8_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    return (np.transpose(image, (1, 2, 0)) * 255.0).round().astype(np.uint8)


def load_recon(path: Path, order: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data = read_pickle(path)
    original = np.asarray(data["x_original"], dtype=np.float32)
    rec_key = "rec_eval_order" if order == "eval" else "rec_attack_order"
    if rec_key not in data:
        raise KeyError(f"{path} does not contain {rec_key}")
    rec = np.asarray(data[rec_key], dtype=np.float32)
    return original, rec, {
        "path": str(path.resolve()),
        "embedding_path": data.get("embedding_path"),
        "embedding_branch": data.get("embedding_branch"),
        "order_key": rec_key,
        "metrics_summary": data.get("metrics_summary"),
    }


def load_branch(
    label: str,
    directory: Path,
    order: str,
    *,
    allow_missing: bool,
) -> Tuple[str, np.ndarray, Dict[str, Any]] | None:
    recon_path = directory / "direct_fia_recon.pkl"
    if not recon_path.is_file():
        if allow_missing:
            return None
        raise FileNotFoundError(
            f"Missing {label} FIA reconstruction: {recon_path}. "
            "Run run_fia_from_embeddings.py for this branch first."
        )
    _original, rec, info = load_recon(recon_path, order)
    info["label"] = label
    return label, rec, info


def make_grid(
    original: np.ndarray,
    branches: List[Tuple[str, np.ndarray, Dict[str, Any]]],
    path: Path,
    *,
    max_items: int,
    item_cols: int,
) -> None:
    n = min([max_items, len(original), *[len(rec) for _label, rec, _info in branches]])
    if n <= 0:
        raise ValueError("No images available for comparison")
    if item_cols <= 0:
        raise ValueError("--item_cols must be >= 1")

    labels = ["original"] + [label for label, _rec, _info in branches]
    stacks = [original] + [rec for _label, rec, _info in branches]
    tile_w = int(original.shape[-1])
    tile_h = int(original.shape[-2])
    label_h = 20
    gutter = 8
    group_cols = len(stacks)
    group_w = group_cols * tile_w
    rows = math.ceil(n / item_cols)
    canvas_w = item_cols * group_w + (item_cols - 1) * gutter
    canvas_h = rows * (tile_h + label_h) + (rows - 1) * gutter
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx in range(n):
        row, item_col = divmod(idx, item_cols)
        x_base = item_col * (group_w + gutter)
        y_base = row * (tile_h + label_h + gutter)
        for col, (label, images) in enumerate(zip(labels, stacks)):
            x = x_base + col * tile_w
            canvas.paste(Image.fromarray(to_uint8_chw(images[idx])), (x, y_base + label_h))
            draw.text((x + 3, y_base + 3), f"{idx}: {label}", fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    naive_dir = resolve_path(args.naive_dir) if args.naive_dir is not None else default_naive_dir(args, run_dir)
    randomized_dir = (
        resolve_path(args.randomized_dir)
        if args.randomized_dir is not None
        else default_randomized_dir(args, run_dir)
    )
    output_path = resolve_path(args.output) if args.output is not None else default_output_path(args, run_dir)

    randomized_recon_path = randomized_dir / "direct_fia_recon.pkl"
    if not randomized_recon_path.is_file():
        raise FileNotFoundError(
            f"Missing randomized FIA reconstruction: {randomized_recon_path}. "
            "Run run_fia_from_embeddings.py for randomized first."
        )
    original, _randomized_probe, randomized_info = load_recon(randomized_recon_path, args.order)
    randomized_info["label"] = "randomized"

    branches: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    naive = load_branch(
        "naive dequant",
        naive_dir,
        args.order,
        allow_missing=args.allow_missing,
    )
    if naive is not None:
        branches.append(naive)
    branches.append(("randomized", _randomized_probe, randomized_info))

    make_grid(
        original,
        branches,
        output_path,
        max_items=args.max_items,
        item_cols=args.item_cols,
    )
    manifest = {
        "schema_version": 1,
        "stage": "fia_comparison_png",
        "output_png": str(output_path.resolve()),
        "order": args.order,
        "max_items": args.max_items,
        "item_cols": args.item_cols,
        "inputs": {
            "original_source": str(randomized_recon_path.resolve()),
            "naive_dir": str(naive_dir.resolve()),
            "randomized_dir": str(randomized_dir.resolve()),
            "branches": [info for _label, _rec, info in branches],
        },
        "note": "Naive dequant means FIA from q*s deterministic dequantized embeddings.",
    }
    manifest_path = output_path.with_suffix(".json")
    write_json(manifest, manifest_path)
    print(f"Comparison PNG saved: {output_path}", flush=True)
    print(f"Comparison manifest saved: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
