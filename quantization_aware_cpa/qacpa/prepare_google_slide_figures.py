from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert current paper figures to 16:9 Google Slides PNGs.")
    parser.add_argument(
        "--component-csv",
        type=Path,
        default=Path("results/v9_stage2_saturation_diagnostics/component_lpips_contributions.csv"),
    )
    parser.add_argument("--input-dir", type=Path, default=Path("results/v9_paper_figures"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v9_google_slides_figures"))
    parser.add_argument("--slide-width", type=int, default=1920)
    parser.add_argument("--slide-height", type=int, default=1080)
    parser.add_argument("--rows-per-slide", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.component_csv)

    make_slide_grid_set(
        rows=rows,
        output_dir=output_dir / "image_grid_source_order",
        slide_width=int(args.slide_width),
        slide_height=int(args.slide_height),
        rows_per_slide=int(args.rows_per_slide),
        sort_by_loss=False,
        title="Original / fp32 / CPA (unmodified) + naive dequant / 4-bit A",
    )
    make_slide_grid_set(
        rows=rows,
        output_dir=output_dir / "image_grid_loss_sorted",
        slide_width=int(args.slide_width),
        slide_height=int(args.slide_height),
        rows_per_slide=int(args.rows_per_slide),
        sort_by_loss=True,
        title="Loss-sorted image comparison",
    )
    make_all_in_one_grid(
        rows=rows,
        output_path=output_dir / "image_grid_all_in_one_source_order_1920x1080.png",
        slide_width=1920,
        slide_height=1080,
        sort_by_loss=False,
        title="All samples: original / fp32 / CPA + naive dequant / A",
    )
    make_all_in_one_grid(
        rows=rows,
        output_path=output_dir / "image_grid_all_in_one_loss_sorted_1920x1080.png",
        slide_width=1920,
        slide_height=1080,
        sort_by_loss=True,
        title="All samples, loss sorted: original / fp32 / CPA + naive dequant / A",
    )
    make_all_in_one_grid(
        rows=rows,
        output_path=output_dir / "image_grid_all_in_one_source_order_3840x2160.png",
        slide_width=3840,
        slide_height=2160,
        sort_by_loss=False,
        title="All samples: original / fp32 / CPA + naive dequant / A",
    )
    make_all_in_one_grid(
        rows=rows,
        output_path=output_dir / "image_grid_all_in_one_loss_sorted_3840x2160.png",
        slide_width=3840,
        slide_height=2160,
        sort_by_loss=True,
        title="All samples, loss sorted: original / fp32 / CPA + naive dequant / A",
    )
    pad_existing_figures(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        slide_width=int(args.slide_width),
        slide_height=int(args.slide_height),
    )
    print(f"wrote Google Slides 16:9 figures to {output_dir}", flush=True)


def make_slide_grid_set(
    *,
    rows: list[dict[str, str]],
    output_dir: Path,
    slide_width: int,
    slide_height: int,
    rows_per_slide: int,
    sort_by_loss: bool,
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["per_sample_loss"])) if sort_by_loss else list(rows)
    total = len(ordered)
    n_pages = (total + rows_per_slide - 1) // rows_per_slide
    for page in range(n_pages):
        start = page * rows_per_slide
        subset = ordered[start : start + rows_per_slide]
        path = output_dir / f"slide_{page + 1:02d}.png"
        make_slide_grid(
            rows=subset,
            output_path=path,
            slide_width=slide_width,
            slide_height=slide_height,
            title=f"{title} ({start + 1}-{start + len(subset)} / {total})",
        )
    write_index(output_dir / "slides.txt", n_pages)


def make_slide_grid(
    *,
    rows: list[dict[str, str]],
    output_path: Path,
    slide_width: int,
    slide_height: int,
    title: str,
) -> None:
    columns = [
        ("Original", "orig_png"),
        ("fp32", "fp32_rec_png"),
        ("4-bit CPA (unmodified)\n+ naive dequant", "naive_rec_png"),
        ("4-bit A", "A_rec_png"),
    ]
    image = Image.new("RGB", (slide_width, slide_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    header_font = load_font(24)
    meta_font = load_font(19)
    small_font = load_font(16)

    margin_x = 56
    top = 42
    title_h = 54
    header_h = 70
    bottom = 36
    label_w = 132
    grid_x = margin_x + label_w
    grid_y = top + title_h + header_h
    grid_w = slide_width - margin_x * 2 - label_w
    grid_h = slide_height - grid_y - bottom
    row_h = grid_h / max(1, len(rows))
    col_w = grid_w / len(columns)
    tile = int(min(col_w - 26, row_h - 16, 248))

    draw.text((margin_x, top), title, fill=(20, 20, 20), font=title_font)
    for col, (label, _key) in enumerate(columns):
        x = int(grid_x + col * col_w + (col_w - tile) / 2)
        draw.text((x, top + title_h + 5), label, fill=(20, 20, 20), font=header_font)

    for row_idx, row in enumerate(rows):
        y0 = int(grid_y + row_idx * row_h)
        center_y = int(y0 + row_h / 2)
        gid = int(row["global_index"])
        loss = float(row["per_sample_loss"])
        source = "retained" if row.get("A_source") == "stage2_retained" else "fallback"
        draw.text((margin_x, center_y - 33), f"g{gid:02d}", fill=(20, 20, 20), font=meta_font)
        draw.text((margin_x, center_y - 8), f"loss {loss:.3g}", fill=(80, 80, 80), font=small_font)
        draw.text((margin_x, center_y + 14), source, fill=(80, 80, 80), font=small_font)
        for col, (_label, key) in enumerate(columns):
            src = resolve_existing_path(row[key], base=Path.cwd())
            if src is None:
                continue
            thumb = Image.open(src).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            x = int(grid_x + col * col_w + (col_w - tile) / 2)
            y = int(center_y - tile / 2)
            image.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def pad_existing_figures(*, input_dir: Path, output_dir: Path, slide_width: int, slide_height: int) -> None:
    names = [
        "lambda_min_over_mp_edge_vs_worst_cosine.png",
        "loss_vs_lpips_scatter_4conditions_fulln.png",
    ]
    for name in names:
        src = input_dir / name
        if not src.is_file():
            continue
        dst = output_dir / f"slide_{src.stem}.png"
        pad_to_slide(src, dst, slide_width=slide_width, slide_height=slide_height)
    schema = input_dir / "m4_heatmap_input_schema.csv"
    if schema.is_file():
        shutil.copyfile(schema, output_dir / schema.name)


def pad_to_slide(src: Path, dst: Path, *, slide_width: int, slide_height: int) -> None:
    canvas = Image.new("RGB", (slide_width, slide_height), "white")
    fig = Image.open(src).convert("RGB")
    max_w = int(slide_width * 0.88)
    max_h = int(slide_height * 0.86)
    scale = min(max_w / fig.width, max_h / fig.height)
    new_size = (int(fig.width * scale), int(fig.height * scale))
    fig = fig.resize(new_size, Image.Resampling.LANCZOS)
    canvas.paste(fig, ((slide_width - new_size[0]) // 2, (slide_height - new_size[1]) // 2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)


def resolve_existing_path(raw: str, *, base: Path) -> Path | None:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_index(path: Path, n_pages: int) -> None:
    lines = [f"slide_{idx + 1:02d}.png" for idx in range(n_pages)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_all_in_one_grid(
    *,
    rows: list[dict[str, str]],
    output_path: Path,
    slide_width: int,
    slide_height: int,
    sort_by_loss: bool,
    title: str,
) -> None:
    ordered = sorted(rows, key=lambda row: float(row["per_sample_loss"])) if sort_by_loss else list(rows)
    columns = [
        ("O", "orig_png"),
        ("F", "fp32_rec_png"),
        ("C", "naive_rec_png"),
        ("A", "A_rec_png"),
    ]
    n_cols = 8
    n_rows = 5
    canvas = Image.new("RGB", (slide_width, slide_height), "white")
    draw = ImageDraw.Draw(canvas)
    scale = slide_width / 1920.0
    title_font = load_font(int(30 * scale))
    label_font = load_font(int(14 * scale))
    mini_font = load_font(int(13 * scale))
    margin_x = int(34 * scale)
    margin_y = int(30 * scale)
    title_h = int(48 * scale)
    block_gap = int(10 * scale)
    block_w = (slide_width - margin_x * 2 - block_gap * (n_cols - 1)) / n_cols
    block_h = (slide_height - margin_y * 2 - title_h - block_gap * (n_rows - 1)) / n_rows
    tile = int(min((block_w - int(18 * scale)) / 2, (block_h - int(42 * scale)) / 2))

    draw.text((margin_x, margin_y), title, fill=(20, 20, 20), font=title_font)
    for idx, row in enumerate(ordered[: n_cols * n_rows]):
        grid_col = idx % n_cols
        grid_row = idx // n_cols
        x0 = int(margin_x + grid_col * (block_w + block_gap))
        y0 = int(margin_y + title_h + grid_row * (block_h + block_gap))
        gid = int(row["global_index"])
        source = "R" if row.get("A_source") == "stage2_retained" else "F"
        loss = float(row["per_sample_loss"])
        draw.text(
            (x0, y0),
            f"g{gid:02d} loss {loss:.3g} {source}",
            fill=(25, 25, 25),
            font=label_font,
        )
        for cond_idx, (short, key) in enumerate(columns):
            src = resolve_existing_path(row[key], base=Path.cwd())
            if src is None:
                continue
            thumb = Image.open(src).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            tx = x0 + (cond_idx % 2) * (tile + int(8 * scale))
            ty = y0 + int(24 * scale) + (cond_idx // 2) * (tile + int(8 * scale))
            canvas.paste(thumb, (tx, ty))
            draw.rectangle((tx, ty, tx + int(17 * scale), ty + int(15 * scale)), fill=(255, 255, 255))
            draw.text((tx + int(2 * scale), ty), short, fill=(0, 0, 0), font=mini_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    main()
