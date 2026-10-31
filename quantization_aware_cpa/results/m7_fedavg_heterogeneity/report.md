# FedAvg reconstruction heterogeneity: existing M7 all80

CPU-only post-hoc descriptive analysis; no new attacks or training. No no-CPA data or LPIPS success threshold.

| Embedding group | n | fp32 LPIPS | dither4 LPIPS | Mean increase | Median ||b_i|| |
|---|---:|---:|---:|---:|---:|
| abs cosine >0.9 | 53 | 0.449383 | 0.460277 | 0.010894 | 0.04090118 |
| abs cosine <=0.9 | 27 | 0.464833 | 0.751555 | 0.286723 | 0.00041611 |

53 embeddings have cosine>0.9,26 have cosine<0.1, and one (g50) has intermediate cosine0.877.26 images cross from high fp32 cosine to low4bit cosine.
The median coefficient norm is 98.29 times larger in the high-cosine group.
Observed update Y approximately equals sum_i b_i z_i^T. Per-row dither amplitude is fixed by the mixture row maximum, so small source contributions face larger relative corruption. Norm/quality association supports this explanation but does not isolate causality, conditioning, source similarity or optimizer effects.
FIA remains imperfect even with accurate features: g48 has source cosine0.999246 but LPIPS0.563253. Image success/failure counts are not inferred from embedding thresholds.

These results describe the previous ported M7 experiment. The running M8 direct-upstream experiment is separate.

![Contribution and reconstruction](contribution_vs_reconstruction.png)
