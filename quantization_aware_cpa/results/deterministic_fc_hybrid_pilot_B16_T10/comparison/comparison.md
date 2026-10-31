Frozen Conv / all FC learning, deterministic uint4, single client, B16/T10.
Same public CPA target, shared 500-step FIA warmup, 50 additional steps per arm.
The added treatment is bin + radius with both image and label gradients; equal iterations, not equal time.

| Arm | PSNR dB | SSIM | VGG LPIPS | Label overlap /16 | Optimization seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| fia | 10.87442 | 0.129483 | 0.683371 | 11 | 4.45 |
| hybrid | 10.85184 | 0.119128 | 0.684643 | 11 | 117.64 |

All16 final images are included; matching is fixed independently per arm using pixel-MSE Hungarian.
No private candidate selection, early stop, or success cutoff. A short single-batch pilot is not a general success/failure claim.
