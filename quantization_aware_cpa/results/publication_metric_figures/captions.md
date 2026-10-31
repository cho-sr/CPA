> **WITHDRAWN / ARCHIVE:** this caption or collection contains claims based on the invalid no-CPA control. Do not use affected figures 2, 3(b), or 5. See [INVALIDATED.md](INVALIDATED.md).

## Figure 1

**Client-update inversion with Cocktail Party Attack and feature inversion.** Cocktail Party Attack (CPA) estimates source embeddings from the first fully connected layer observation; feature inversion attack (FIA) reconstructs images from those embeddings. A pretrained VGG-16 with frozen convolutional features is evaluated on ImageNet. Five independent client updates reset the model to round 0 and use 16 images each, ten FC-only SGD steps (learning rate 0.001), and the same 80 images in both precision conditions. The FP32 attack observes the actual parameter difference divided by the learning rate. The 4-bit condition uses the existing stochastic per-row subtractive-dither observation model, not an evaluated integer transmission codec. Both use all 16 CPA components and unchanged FIA (25,000 iterations, TV=0.1); no ground-truth embeddings supplement missing sources. (a–c) Small translucent points are individual image metrics, open circles are the five client-batch means, gray lines connect the same client batch across conditions, and black diamonds are means over all 80 images. These marks do not represent confidence intervals. (d) Empirical cumulative distributions of matched absolute source cosine; the dashed line is the strict recovery threshold of 0.9. Counts of 79/80 and 53/80 measure embedding recovery, not an image-level LPIPS success rule. The paired mean LPIPS increase is 0.10398628 (4-bit observation model minus FP32). Full source values and image metrics accompany the figure.

## Figure 2

**Contribution of source separation to gradient inversion.** Cocktail Party Attack followed by feature inversion (CPA + FIA) is compared with FIA applied directly to fixed observed gradient rows and FIA applied to observation-independent random embeddings. All methods use pretrained round-0 VGG-16, ImageNet, victim batch size 8, and the existing FIA budget (25,000 iterations, TV=0.1, Adam learning rate 0.1). The direct-row control permits row L2 normalization only, including selected zero rows; it omits CPA source separation. Random-embedding FIA uses normalized IID uniform targets and is not a TV-only ablation. In the FP32 panels (a–c), the four historical ground-truth embedding supplements at global indices 22, 24, 32 and 36 are excluded from all three methods, leaving the same 36 evaluation images. In the 4-bit panels (d–f), all methods use the same 40 images. Consequently the rows have different sample sets and are not a matched precision-effect comparison. The 4-bit CPA pipeline applies unchanged CPA + FIA to the per-row subtractive-dither observation model; this is not a validated integer transmission codec. Small translucent points show individual images, open circles show five victim-batch means, gray lines join the same batch across methods, and diamonds show image-weighted means. In the 36-image panels the batch sizes are 8, 8, 7, 7 and 6, so the diamond is not an unweighted average of the five batch means. The marks are descriptive, not confidence intervals. The control metrics use evaluation-only within-batch permutation matching after FIA; no sign or scale correction is applied to control targets. Historical FP32 FIA invocation sizes differ from those of the controls. The original prespecified confirmatory LPIPS comparison uses 40 images in the 4-bit condition, enumerates all 32 sign flips of five batch-mean differences, and assumes sign symmetry. Both control comparisons have positive effects and one-sided p=0.03125; their intersection-union p is 0.03125. No additional significance test is implied for PSNR, SSIM or the FP32 sensitivity panels.

## Figure 3

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

## Figure 4

Figure 4. Source-count prediction and the association between training loss and reconstruction quality.

**(a)** Predicted versus measured recoverable source count across all 96 stored diagnostic conditions, with different markers for FedSGD (T = 1, 48 conditions) and FedAvg (T = 10, 48 conditions). The grid includes client sizes 16, 32 and 64, bit widths 2, 3, 4 and 8, and per-row, per-bucket and per-tensor scale conditions; it is not a fully crossed factorial design across all axes. Measured counts use matched source absolute cosine strictly greater than 0.9. The dashed line denotes exact prediction. Coordinates are the original integer counts: coincident conditions overlap, with no jitter or omitted rows. Predictions are exact for **81/96 conditions**, with mean absolute error **0.218750 sources** (displayed as 0.219) and maximum absolute error 2. The complete T = 1 subset is exact in 39/48 conditions, and the complete T = 10 subset in 42/48. These mechanism diagnostics use the archived factorized cumulative-gradient observations; they are distinct from reconstruction experiments using actual FP32 client parameter differences.

**(b)** Per-image cross-entropy loss versus LPIPS for gradient CPA + FIA on VGG-16/ImageNet (FedSGD, client size 8). Every included image is plotted. FP32 contains **36 images**, excluding the four historical oracle-supplemented reconstructions (global indices 22, 24, 32 and 36); the 4-bit observation model contains **all 40 images**, with no oracle supplementation. The horizontal axis is logarithmic; correlations are calculated from the original values. Spearman correlation is **rho = +0.030888030888** for FP32 (two-sided asymptotic p = 0.858071088545) and **rho = -0.590619136961** for the 4-bit observation model (p = 6.02093714009e-05). These are descriptive image-level correlations from different sample scopes; the displayed coefficients are not a statistical test of their difference or evidence of a causal precision effect. No regression line or causal trend is fitted.

The 4-bit condition retains the repository's stochastic per-row subtractive-dither observation model, Y_hat = Y + Uniform(-delta/2, delta/2), with delta = 2 max(abs(row)) / (2^4 - 1). It does not evaluate an integer transmission codec. FP32 has no applicable quantization scale granularity. LPIPS uses the existing VGG perceptual-distance implementation; lower is better.

All plotted values and source-row provenance are supplied in [fig04_data.csv](fig04_data.csv). The figure is reproduced without model execution by [render_prediction_diagnostics.py](render_prediction_diagnostics.py). Exports use DejaVu Sans, a white background, 300 dpi PNG and vector PDF/SVG.

## Figure 5

**Descriptive LPIPS normalization and fixed-cutoff image counts.** (a) The relative improvement of CPA + FIA under the 4-bit per-row subtractive-dither observation model is defined as 100(F-L)/(F-P), where F is mean LPIPS for FIA on the observed 4-bit gradient rows, P is the FP32 CPA + FIA mean, and L is the 4-bit CPA + FIA mean. The reported-set analysis uses F=0.7538226299 over 40 images, P=0.4464729486 over 36 images (four ground-truth embedding supplements excluded), and L=0.5439403489 over 40 images, giving 68.2878%. The matched-36 sensitivity recomputes all three quantities on the same 36 images (F=0.7519503211, P=0.4464729486, L=0.5176888704), giving 76.6870%. The dashed 100% line denotes the corresponding FP32 reference. This normalization is descriptive, not an information-theoretic recovery fraction or a universally calibrated signal-retention rate. (b) The unchanged follow-up rule counts images with LPIPS strictly below 0.570428904891014, the linear-interpolated fifth percentile of the 40-image observed-gradient-row FIA control under the 4-bit observation model. This numeric threshold was recorded before applying it to candidates. Counts are 36/36 for FP32 gradient inversion, 29/40 for 4-bit gradient inversion, and 79/80 for FP32 client-update inversion. They are descriptive lower-tail counts, not individual-image significance tests or the |source cosine|>0.9 embedding criterion. The client-update application transfers a gradient-derived threshold and is exploratory. The 4-bit client-update condition is not part of this archived fixed-cutoff analysis. The dated follow-up amendment was fixed after historical results were available; it is not a retroactive preregistration. These counts must not be mixed with the earlier precision-specific leave-batch-out rank counts. No error bars or additional tests are inferred.
