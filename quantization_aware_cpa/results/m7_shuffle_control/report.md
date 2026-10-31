# M7 shuffled pairing control

Each reconstruction is paired to a uniformly permuted original within its complete evaluation set; no rematching.
NumPy default_rng seeds 0–19; ordinary permutations retain and report fixed points. GIA fp32 excludes oracle-fill gids 22, 24, 32, 36.

| Condition | N | Matched LPIPS | Shuffled mean | SD of 20 seed means | Seed-mean q05 / q95 | Change |
|---|---:|---:|---:|---:|---:|---:|
| gia_fp32 | 36 | 0.446473 | 0.808424 | 0.011869 | 0.792538 / 0.824281 | +0.361951 |
| gia_4bit | 40 | 0.543940 | 0.798902 | 0.012408 | 0.781252 / 0.810462 | +0.254962 |
| uia_fp32 | 80 | 0.454597 | 0.816526 | 0.005440 | 0.807075 / 0.822743 | +0.361929 |

All three shuffled means are above the no-CPA 4-bit reference 0.753823. GIA 4-bit increases from 0.543940 to 0.798902 (+0.254962), so its good matched LPIPS depends on correct correspondence rather than merely the image prior.
Sample SD uses ddof=1; quantiles use linear interpolation. Pooled pair SD/quantiles are separately stored in summary.csv / summary.json.
Twenty seeds reuse the same images; the 720 / 800 / 1600 pairs are not independent image samples. Fixed-point counts and every endpoint ID are retained in per_image_metrics.csv.
No-CPA and prior-only assess FIA without recovered observation structure; shuffled pairing assesses loss of the correct reconstruction-to-original correspondence.
A shuffled mean need not equal the no-CPA mean: generated-image appearance differs. A large rise from matched LPIPS establishes sensitivity to correspondence; it does not by itself isolate the CPA contribution.
