from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quantization_aware_cpa/results/m6_uia_fp32_B16_fia"
NAIVE = ROOT / "quantization_aware_cpa/results/m8_fedavg_naive_cpa_dither4_B16/all80_rec.pt"
TOPK50 = ROOT / "quantization_aware_cpa/results/cpa_fia_topk50_same80_B16/all80_rec.pt"
OUT = ROOT / "quantization_aware_cpa/exports/naive_vs_topk50_comparison/original_naive_topk50_all80.png"


def load_font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def rgb(tensor: torch.Tensor) -> Image.Image:
    if tuple(tensor.shape) != (3, 224, 224):
        raise RuntimeError(f"Unexpected image shape: {tuple(tensor.shape)}")
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    originals = []
    for batch in range(5):
        inputs = torch.load(SOURCE / f"batch{batch:02d}/inputs.pt", map_location="cpu", weights_only=False)
        x = inputs["x"].detach().cpu()
        mean = x.new_tensor((.485, .456, .406))[None, :, None, None]
        std = x.new_tensor((.229, .224, .225))[None, :, None, None]
        originals.append((x * std + mean).clamp(0, 1))
    originals = torch.cat(originals)
    naive = torch.load(NAIVE, map_location="cpu", weights_only=True)
    topk50 = torch.load(TOPK50, map_location="cpu", weights_only=True)
    if any(tuple(t.shape) != (80, 3, 224, 224) for t in (originals, naive, topk50)):
        raise RuntimeError("Expected 80 native-resolution RGB images in every arm")

    tile, gap, header, labels, columns = 224, 12, 160, 58, 5
    block = 3 * tile + 2 * gap
    width = gap + columns * (block + gap)
    height = header + ((80 + columns - 1) // columns) * (tile + labels + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), "Original → Naive dequantization → Top-k 50% sparsification", fill="#0f172a", font=load_font(32, True))
    draw.text((gap, 55), "Same 80 images, original order | each block is Original / Naive / Top-k 50%", fill="#334155", font=load_font(22))
    draw.text((gap, 94), "Naive: LPIPS 0.5561, PSNR 21.937, SSIM 0.2980 | Top-k 50%: LPIPS 0.4864, PSNR 22.288, SSIM 0.3344", fill="#475569", font=load_font(19))

    titles = ("Original", "Naive", "Top-k 50%")
    arms = (originals, naive, topk50)
    for index in range(80):
        col, row = index % columns, index // columns
        left = gap + col * (block + gap)
        top = header + row * (tile + labels + gap)
        draw.text((left, top), f"image {index:02d}", fill="#0f172a", font=load_font(17, True))
        for panel, (title, arm) in enumerate(zip(titles, arms)):
            x = left + panel * (tile + gap)
            draw.text((x, top + 25), title, fill="#334155", font=load_font(16))
            canvas.paste(rgb(arm[index]), (x, top + labels))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, format="PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
