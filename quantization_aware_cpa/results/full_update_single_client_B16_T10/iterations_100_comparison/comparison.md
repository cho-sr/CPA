Same batch00, VGG16 Conv+FC, B16/T10, 4bit observations; only outer iterations increase from 3 to 100.
Fresh identical seeded runs; no pilot optimizer state is resumed. All 16 final images are evaluated.

| Arm | Iterations | PSNR (dB) | SSIM | Label accuracy | Hard closure fit | Bin violation rate | Time (s) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| cell | 3 | 7.48790 | 0.008083 | 0.000 | False | 0.224134 | 47.79 |
| center | 3 | 7.48856 | 0.008089 | 0.000 | False | 0.227115 | 48.05 |
| cell | 100 | 6.08976 | 0.005472 | 0.000 | False | 0.253444 | 1502.95 |
| center | 100 | 6.02765 | 0.005649 | 0.000 | False | 0.213540 | 1501.64 |

The recorded fingerprints match, and the first three numerical logs match each pilot exactly.
Convergence plots normalize each different soft objective by its own initial value; points are pre-update.
Final hard-label fit and offline image metrics refer to the saved post-update candidate.
No reconstruction-success threshold or statistical superiority is established by this single-batch comparison.

Descriptive VGG LPIPS uses the existing Hungarian pairs and all 16 images; no success cutoff is applied.

| Arm | Mean VGG LPIPS |
| --- | ---: |
| cell_3 | 0.839900 |
| center_3 | 0.839646 |
| cell_100 | 0.864648 |
| center_100 | 0.871043 |
