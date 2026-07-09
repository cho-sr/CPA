#!/usr/bin/env python3
#
# Export Cocktail Party Attack reconstruction pickles to PNG files.

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


XSHAPE_DICT = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "tiny_imagenet": (3, 64, 64),
    "imagenet": (3, 224, 224),
}


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def to_nchw(array, ds):
    array = np.asarray(array)
    if array.ndim == 4:
        return array

    if array.ndim == 2:
        c, h, w = XSHAPE_DICT[ds]
        return array.reshape(array.shape[0], c, h, w)

    raise ValueError(f"Expected image array with 2 or 4 dims, got shape {array.shape}")


def image_from_chw(chw, scale_each=False):
    chw = np.asarray(chw, dtype=np.float32)

    if scale_each:
        min_val = float(chw.min())
        max_val = float(chw.max())
        if max_val > min_val:
            chw = (chw - min_val) / (max_val - min_val)

    chw = np.clip(chw, 0.0, 1.0)
    hwc = np.transpose(chw, (1, 2, 0))
    return Image.fromarray((hwc * 255.0).round().astype(np.uint8), mode="RGB")


def make_grid(images, nrow, pad=2, bg=(255, 255, 255)):
    if not images:
        raise ValueError("No images to save")

    width, height = images[0].size
    nrow = max(1, min(nrow, len(images)))
    ncol = int(math.ceil(len(images) / nrow))

    canvas_w = nrow * width + (nrow - 1) * pad
    canvas_h = ncol * height + (ncol - 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        canvas.paste(img, (col * (width + pad), row * (height + pad)))

    return canvas


def make_compare_images(inp, rec, scale_each=False, label=True):
    images = []
    for idx in range(inp.shape[0]):
        orig_img = image_from_chw(inp[idx], scale_each=scale_each)
        rec_img = image_from_chw(rec[idx], scale_each=scale_each)

        w, h = orig_img.size
        label_h = 14 if label else 0
        canvas = Image.new("RGB", (w * 2 + 2, h + label_h), (255, 255, 255))
        canvas.paste(orig_img, (0, label_h))
        canvas.paste(rec_img, (w + 2, label_h))

        if label:
            draw = ImageDraw.Draw(canvas)
            draw.text((2, 1), f"orig {idx}", fill=(0, 0, 0))
            draw.text((w + 4, 1), "rec", fill=(0, 0, 0))

        images.append(canvas)

    return images


def save_batch(rec_dict, batch_idx, out_dir, ds, nrow, max_images, scale_each, save_individual):
    inp = to_nchw(rec_dict["inp"][batch_idx], ds)
    rec = to_nchw(rec_dict["rec"][batch_idx], ds)

    if max_images is not None:
        inp = inp[:max_images]
        rec = rec[:max_images]

    out_dir.mkdir(parents=True, exist_ok=True)

    orig_images = [image_from_chw(img, scale_each=scale_each) for img in inp]
    rec_images = [image_from_chw(img, scale_each=scale_each) for img in rec]
    compare_images = make_compare_images(inp, rec, scale_each=scale_each)

    make_grid(orig_images, nrow=nrow).save(out_dir / f"batch_{batch_idx:03d}_original.png")
    make_grid(rec_images, nrow=nrow).save(out_dir / f"batch_{batch_idx:03d}_reconstruction.png")
    make_grid(compare_images, nrow=max(1, nrow // 2)).save(
        out_dir / f"batch_{batch_idx:03d}_compare.png"
    )

    if save_individual:
        individual_dir = out_dir / f"batch_{batch_idx:03d}_individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for idx, (orig_img, rec_img) in enumerate(zip(orig_images, rec_images)):
            orig_img.save(individual_dir / f"{idx:03d}_original.png")
            rec_img.save(individual_dir / f"{idx:03d}_reconstruction.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export attack *_rec.pkl files to PNG grids."
    )
    parser.add_argument("--rec_file", required=True, help="Path to *_rec.pkl file")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: <rec_file parent>/png",
    )
    parser.add_argument(
        "--ds",
        default="tiny_imagenet",
        choices=sorted(XSHAPE_DICT.keys()),
        help="Dataset shape to use if arrays are flattened",
    )
    parser.add_argument(
        "--batch",
        default="all",
        help="Batch index to export, or 'all'. Default: all",
    )
    parser.add_argument("--nrow", type=int, default=8, help="Images per grid row")
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum images per batch to export",
    )
    parser.add_argument(
        "--scale_each",
        action="store_true",
        help="Rescale each image independently to 0..1 before saving",
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Also save each original/reconstruction image separately",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rec_file = Path(args.rec_file)
    out_dir = Path(args.out_dir) if args.out_dir else rec_file.parent / "png"

    rec_dict = load_pickle(rec_file)
    if "inp" not in rec_dict or "rec" not in rec_dict:
        raise ValueError("rec_file must contain 'inp' and 'rec' keys")

    n_batches = len(rec_dict["rec"])
    if args.batch == "all":
        batch_indices = range(n_batches)
    else:
        batch_idx = int(args.batch)
        if batch_idx < 0 or batch_idx >= n_batches:
            raise ValueError(f"Batch index {batch_idx} out of range 0..{n_batches - 1}")
        batch_indices = [batch_idx]

    for batch_idx in batch_indices:
        save_batch(
            rec_dict=rec_dict,
            batch_idx=batch_idx,
            out_dir=out_dir,
            ds=args.ds,
            nrow=args.nrow,
            max_images=args.max_images,
            scale_each=args.scale_each,
            save_individual=args.save_individual,
        )

    print(f"Saved PNG files to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
