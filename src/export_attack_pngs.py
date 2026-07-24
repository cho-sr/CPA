#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


def _to_bchw(value, name):
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape NCHW or NHWC; got {tuple(tensor.shape)}")
    if tensor.shape[1] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[1] not in (1, 3):
        raise ValueError(f"{name} must be image-like; got {tuple(tensor.shape)}")
    return tensor.clamp(0, 1)


def _tensor_to_pil(image, cell_size):
    image = image.clamp(0, 1)
    if image.shape[0] == 1:
        array = (image[0] * 255).byte().numpy()
        pil = Image.fromarray(array, mode="L").convert("RGB")
    else:
        array = (image.permute(1, 2, 0) * 255).byte().numpy()
        pil = Image.fromarray(array, mode="RGB")

    if cell_size > 0 and pil.size != (cell_size, cell_size):
        pil = pil.resize((cell_size, cell_size), Image.Resampling.NEAREST)
    return pil


def _draw_centered(draw, box, text, fill):
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - text_w) / 2, y0 + (y1 - y0 - text_h) / 2),
        text,
        fill=fill,
        font=font,
    )


def make_pair_grid(original, reconstruction, max_images, pairs_per_row, cell_size):
    original = _to_bchw(original, "original")
    reconstruction = _to_bchw(reconstruction, "reconstruction")
    n_images = min(max_images, original.shape[0], reconstruction.shape[0])
    if n_images == 0:
        raise ValueError("No images available to export.")

    sample = _tensor_to_pil(original[0], cell_size)
    cell_w, cell_h = sample.size
    label_h = 18
    pad = 10
    gap = 6
    pair_gap = 14
    pairs_per_row = max(1, pairs_per_row)
    n_rows = (n_images + pairs_per_row - 1) // pairs_per_row
    pair_w = cell_w * 2 + gap
    row_h = label_h + cell_h
    canvas_w = pad * 2 + pairs_per_row * pair_w + (pairs_per_row - 1) * pair_gap
    canvas_h = pad * 2 + n_rows * row_h + (n_rows - 1) * pair_gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx in range(n_images):
        row = idx // pairs_per_row
        col = idx % pairs_per_row
        x = pad + col * (pair_w + pair_gap)
        y = pad + row * (row_h + pair_gap)
        orig_img = _tensor_to_pil(original[idx], cell_size)
        rec_img = _tensor_to_pil(reconstruction[idx], cell_size)

        _draw_centered(draw, (x, y, x + cell_w, y + label_h), "original", (40, 40, 40))
        _draw_centered(
            draw,
            (x + cell_w + gap, y, x + pair_w, y + label_h),
            "reconstruction",
            (40, 40, 40),
        )
        canvas.paste(orig_img, (x, y + label_h))
        canvas.paste(rec_img, (x + cell_w + gap, y + label_h))

        draw.rectangle((x, y + label_h, x + cell_w - 1, y + label_h + cell_h - 1), outline=(210, 210, 210))
        draw.rectangle(
            (
                x + cell_w + gap,
                y + label_h,
                x + pair_w - 1,
                y + label_h + cell_h - 1,
            ),
            outline=(210, 210, 210),
        )

    return canvas


def resolve_rec_file(args):
    if args.rec_pkl is not None:
        return Path(args.rec_pkl)
    if args.exp_path is None or args.batch_size is None:
        raise SystemExit("Use --rec_pkl, or use --exp_path together with --batch_size.")
    return Path(args.exp_path) / f"{args.batch_size}_rec.pkl"


def main():
    parser = argparse.ArgumentParser(description="Export CPA attack reconstructions as PNGs.")
    parser.add_argument("--rec_pkl", type=str, default=None, help="Path to *_rec.pkl.")
    parser.add_argument("--exp_path", type=str, default=None, help="Attack experiment directory.")
    parser.add_argument("--batch_size", type=int, default=None, help="Attack batch size.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--max_images", type=int, default=16, help="Images per batch to export.")
    parser.add_argument("--pairs_per_row", type=int, default=4, help="Original/reconstruction pairs per row.")
    parser.add_argument("--cell_size", type=int, default=128, help="Rendered square image size. Use 0 for native size.")
    args = parser.parse_args()

    rec_file = resolve_rec_file(args)
    if not rec_file.exists():
        raise FileNotFoundError(rec_file)

    out_dir = Path(args.out_dir) if args.out_dir is not None else rec_file.parent / "png"
    out_dir.mkdir(parents=True, exist_ok=True)

    with rec_file.open("rb") as handle:
        rec_data = pickle.load(handle)

    if "inp" not in rec_data or "rec" not in rec_data:
        raise KeyError(f"{rec_file} must contain 'inp' and 'rec' entries.")

    written = []
    for batch_idx, (original, reconstruction) in enumerate(zip(rec_data["inp"], rec_data["rec"])):
        grid = make_pair_grid(
            original,
            reconstruction,
            max_images=args.max_images,
            pairs_per_row=args.pairs_per_row,
            cell_size=args.cell_size,
        )
        out_file = out_dir / f"batch_{batch_idx:03d}_original_reconstruction_pairs.png"
        grid.save(out_file)
        written.append(out_file)

    print(f"Exported {len(written)} comparison PNG(s) to {out_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
