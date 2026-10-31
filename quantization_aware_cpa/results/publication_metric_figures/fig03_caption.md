> **WITHDRAWN / ARCHIVE:** this caption or collection contains claims based on the invalid no-CPA control. Do not use affected figures 2, 3(b), or 5. See [INVALIDATED.md](INVALIDATED.md).

# Figure 3. Image-assignment sensitivity and empirical reconstruction references

**(a)** Correctly assigned reconstructions from the cocktail party attack (CPA)
followed by feature inversion (FIA), compared with 20 fixed random permutations
of their image assignments. Each orange open circle is the mean LPIPS for one
permutation; the orange diamond is the mean across the 20 permutations. Blue
circles are the original, correctly assigned means. Gradient observations use
batch size 8: FP32 includes only the 36 CPA-derived targets after excluding the
four oracle-target supplements, and the 4-bit observation model includes all 40 images.
Client-update observations use 16 images per independent round-0 update, 10
local steps, and 80 images in total. These are different experimental settings,
so comparisons across rows do not isolate a gradient-versus-update effect.
Permutations retain fixed points and are not rematched; the 720, 800, and 1,600
pairs reuse 36, 40, and 80 originals, respectively. The plotted permutation means
describe sensitivity to assignment, not independent replications or confidence
intervals. Connecting segments compare assignment conditions within each row.

**(b)** LPIPS for FIA applied to preselected observed gradient rows (FP32 and
4-bit observation model), FIA applied to independent random embeddings, and three image
references. Blue circles are means over the same 40 gradient-victim images;
gray open circles are the five victim-batch means (eight images each). No
confidence intervals or significance tests are inferred from the point spread.
Observed rows receive only row L2 normalization; selected zero rows are
retained. Random embedding targets are sampled independently of observations
and true embeddings and are row-normalized. The FIA controls use the original
25,000-iteration objective and budget. Their primary scores use an oracle,
within-batch, absolute-embedding-cosine assignment after FIA; this permutation
does not change the signed targets supplied to FIA. Historical CPA target
alignment is preserved. Thus excluding oracle supplements does not mean
evaluation is free of known-reference source matching.

The full-validation mean image averages all 50,000 transformed ImageNet
validation images, gray is RGB 0.5, and uniform random images are sampled from
Uniform[0,1). These references are empirical controls, not mathematical LPIPS
floors. The 4-bit condition uses the existing stochastic per-row
subtractive-dither observation model, not an integer transmission-codec test.
LPIPS uses the VGG backbone and is lower for closer image pairs. Both panels
use the same horizontal scale. Marker displacement in the vertical direction
only separates overlapping points. Numbers above the main markers are rounded
to three decimals; full precision and every displayed replicate are in
`fig03_data.csv`. No published GMA reference, hybrid FP32 mean, or reconstruction
success count is included in this figure.

## Sources

The figure reads saved CSVs only. No training, reconstruction, or LPIPS
re-evaluation was performed. Scientific scope follows
`codex/docs/QACPA_SPEC.md` section 0.8 and
`codex/docs/M7_FOLLOWUP_ANALYSIS_PROTOCOL.json`.

- `quantization_aware_cpa/results/m7_shuffle_control/summary.csv` — SHA-256 `cc5665c565f75687dcb4a64ce4f23bdc073cc92d767727bcc30f4836bbef08a1`
- `quantization_aware_cpa/results/m7_shuffle_control/per_seed_summary.csv` — SHA-256 `1fd40fda02e35dc3f9d1850b99dde023566a583da78ba940ca19a3b7617a1fb6`
- `quantization_aware_cpa/results/m7_normalized_metrics/normalized_metrics.csv` — SHA-256 `c8e463ddd9e1a193bb1997f7b7ee933101a702187210b544d016c20ac1f9c438`
- `quantization_aware_cpa/results/m7_no_cpa_control/fp32/per_image_metrics.csv` — SHA-256 `4a62ff5ac976296d8cb2bfe19de6bab60b0fa0b40e78ddc6f575d25479211167`
- `quantization_aware_cpa/results/m7_no_cpa_control/4bit/per_image_metrics.csv` — SHA-256 `329b0363f05667423e412df738e50cf8dc815264750b558f98cf277f48a9b026`
- `quantization_aware_cpa/results/m7_prior_only/prior_only/per_image_metrics.csv` — SHA-256 `7bf30747b6aa01df6fe3b6386d8e7f0df6b96ce26c2d4c070fae2fd6ba641077`
- `quantization_aware_cpa/results/m7_recompute_controls/per_image_metrics.csv` — SHA-256 `1bf3b8af8b0cab17a3301535d71666854cac597d84bba980035e721b22c2b35c`
