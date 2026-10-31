from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
NAIVE = ROOT / "quantization_aware_cpa/results/m8_fedavg_naive_cpa_dither4_B16/all80_reconstruction_landscape.png"
TOPK50 = ROOT / "quantization_aware_cpa/exports/topk_same80_reconstruction_16x9/topk50_original_vs_reconstruction_16x9.png"
OUT = ROOT / "quantization_aware_cpa/exports/naive_vs_topk50_comparison/naive_vs_topk50_all80_16x9.png"


def font(size: int, bold: bool = False):
    names = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def main():
    if not NAIVE.exists() or not TOPK50.exists():
        raise FileNotFoundError("Both source landscape PNGs are required")
    canvas = Image.new("RGB", (3840, 2160), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 45), "Naive dequantization vs Top-k 50% sparsification", fill="black", font=font(64, True))
    draw.text((80, 130), "Same 80-image evaluation set | lower LPIPS, higher PSNR/SSIM are better", fill=(70, 70, 70), font=font(34))

    panel_w, panel_h = 1780, 1001
    y = 300
    for x, source, title, subtitle in [
        (100, NAIVE, "Naive dequantization", "LPIPS 0.5561 | PSNR 21.937 | SSIM 0.2980"),
        (1960, TOPK50, "Top-k 50% sparsification", "LPIPS 0.4864 | PSNR 22.288 | SSIM 0.3344"),
    ]:
        draw.text((x, 195), title, fill="black", font=font(46, True))
        draw.text((x, 250), subtitle, fill=(50, 50, 50), font=font(30))
        with Image.open(source) as image:
            image = image.convert("RGB")
            image = ImageOps.contain(image, (panel_w, panel_h), Image.Resampling.LANCZOS)
            px = x + (panel_w - image.width) // 2
            py = y + (panel_h - image.height) // 2
            canvas.paste(image, (px, py))
            draw.rectangle((x, y, x + panel_w, y + panel_h), outline=(120, 120, 120), width=3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, format="PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
