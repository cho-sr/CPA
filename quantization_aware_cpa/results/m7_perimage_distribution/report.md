# M7 per-image LPIPS and descriptive success counts

Frozen threshold: strict LPIPS < **0.5704289049**, the linear lower fifth percentile of all 40 GIA 4-bit no-CPA LPIPS values.
The control-only numerical threshold was saved before candidate counting. This successor uses one common descriptive cutoff; it does not replace or amend the historical M7 preregistered rule.
UIA counts apply the GIA control reference across protocols and are exploratory. These are not individual-image significance decisions.

| Condition | Mean LPIPS | Success count |
|---|---:|---:|
| GIA fp32 (36; no oracle fills) | 0.446473 | 36/36 |
| GIA 4-bit (40) | 0.543940 | 29/40 |
| UIA fp32 (80) | 0.454597 | 79/80 |
| Shuffled GIA fp32 (20 x 36) | 0.808424 | 27/720 |
| Shuffled GIA 4-bit (20 x 40) | 0.798902 | 20/800 |
| Shuffled UIA fp32 (20 x 80) | 0.816526 | 19/1600 |
| Prior-only (40) | 0.773775 | 0/40 |
| No-CPA 4-bit (40; threshold reference) | 0.753823 | 2/40 |
| No-CPA fp32 (40) | 0.756820 | 10/40 |
| Constant: full-val mean (40) | 0.805476 | 0/40 |
| Constant: gray 0.5 (40) | 0.829199 | 0/40 |
| Constant control: uniform noise (40) | 0.856640 | 0/40 |

Shuffled denominators count pair evaluations across 20 seeds, not independent images; per-seed image counts are in shuffled_per_seed_success_counts.csv.
Every below-threshold shuffled pair is a retained fixed point; zero nonfixed shuffled pairs pass the threshold in any of the three conditions.
GIA 4-bit has two visible LPIPS groups, with KDE modes near 0.45 and 0.78 across all three preregistered bandwidths. This supports heterogeneous reconstruction quality in this sample. However, 29/40 pass the frozen cutoff: a claim of 'a minority of recoveries and a majority of failures' is inconsistent with these counts.
No-CPA fp32 and 4-bit means are close (0.756820 / 0.753823), but their distributions differ: SD 0.186229 / 0.109000 and common-cutoff counts 10/40 / 2/40. Mean stability does not establish distributional equivalence.
![Common-axis distributions](lpips_histogram_overlay.png)

![4-bit shape sensitivity](gia_4bit_modality_sensitivity.png)

The four exploratory histogram variants and three preregistered KDE bandwidths disclose shape sensitivity at N=40. No formal multimodality test is claimed. A smooth unimodal curve does not establish identical degradation for every image, and incidental histogram peaks do not establish a minority-only recovery mixture.
