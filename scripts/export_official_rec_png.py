#!/usr/bin/env python3
import argparse
import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def to_uint8_grid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW array, got shape {x.shape}")
    if x.shape[1] == 3:
        x = np.transpose(x, (0, 2, 3, 1))
    elif x.shape[-1] != 3:
        raise ValueError(f"Expected channel dimension of 3, got shape {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def make_grid(images: np.ndarray, cols: int = 8, gap: int = 4) -> Image.Image:
    images = to_uint8_grid(images)
    n, h, w, c = images.shape
    rows = math.ceil(n / cols)
    canvas = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), "white")
    for idx, img in enumerate(images):
        y, x = divmod(idx, cols)
        canvas.paste(Image.fromarray(img, "RGB"), (x * (w + gap), y * (h + gap)))
    return canvas


def make_comparison(inp: np.ndarray, rec: np.ndarray, cols: int = 4, gap: int = 4, label_h: int = 22) -> Image.Image:
    inp = to_uint8_grid(inp)
    rec = to_uint8_grid(rec)
    n = min(len(inp), len(rec))
    h, w = inp.shape[1], inp.shape[2]
    cell_w = (2 * w) + gap
    rows = math.ceil(n / cols)
    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols - 1) * gap, rows * (h + label_h) + (rows - 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for idx in range(n):
        row, col = divmod(idx, cols)
        x0 = col * (cell_w + gap)
        y0 = row * (h + label_h + gap)
        draw.text((x0, y0), f"{idx:02d} original", fill=(0, 0, 0))
        draw.text((x0 + w + gap, y0), f"{idx:02d} recovered", fill=(0, 0, 0))
        canvas.paste(Image.fromarray(inp[idx], "RGB"), (x0, y0 + label_h))
        canvas.paste(Image.fromarray(rec[idx], "RGB"), (x0 + w + gap, y0 + label_h))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rec_pkl", type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--save_individual", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.rec_pkl.with_suffix("").parent / "png"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.rec_pkl.stem

    with args.rec_pkl.open("rb") as f:
        data = pickle.load(f)

    inp = np.concatenate(data["inp"], axis=0)
    rec = np.concatenate(data["rec"], axis=0)

    rec_grid = args.out_dir / f"{prefix}_recovered_grid.png"
    cmp_grid = args.out_dir / f"{prefix}_comparison_grid.png"
    make_grid(rec, cols=args.cols).save(rec_grid)
    make_comparison(inp, rec, cols=max(1, args.cols // 2)).save(cmp_grid)

    if args.save_individual:
        rec_uint8 = to_uint8_grid(rec)
        indiv_dir = args.out_dir / f"{prefix}_recovered"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(rec_uint8):
            Image.fromarray(img, "RGB").save(indiv_dir / f"{idx:02d}.png")

    print(f"Saved {rec_grid}")
    print(f"Saved {cmp_grid}")
    if args.save_individual:
        print(f"Saved individual images under {indiv_dir}")


if __name__ == "__main__":
    main()
