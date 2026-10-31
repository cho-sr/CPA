# UIA / FedAvg Published-Protocol Comparison Note

Purpose: make the later FIA/LPIPS comparison against CPA Appendix B Table 5
auditable before running image reconstruction.

Current reporting amendment (2026-09-09): UIA fp32 n16/T10 all80 has LPIPS
**0.4546**, PSNR **22.406**, SSIM **0.3427**, embedding |cos|>0.9 **79/80**,
and no oracle fills. The published CPA+FIA range over batch16-128 is
**0.410-0.448**; the result is **0.0066 above its upper endpoint**, a
cross-batch range-boundary comparison. Current GIA fp32 is **0.4465 over36
images**, recalculated without the four oracle supplements. Section 0.8 of
QACPA_SPEC.md and M7_FOLLOWUP_ANALYSIS_PROTOCOL.json authorize the new
saved-update-only UIA4bit follow-up. The original fp32 registration below is
historical and its immutable config/lock are preserved.

## Published CPA Reference

Primary paper: Kariyappa et al., "Cocktail Party Attack: Breaking
Aggregation-Based Privacy in Federated Learning using Independent Component
Analysis", ICML 2023 / PMLR v202.

Relevant reference points:

- The paper attacks gradients / weight updates aggregated over many samples
  using the FC-layer linear factorization.
- Appendix B Table 5 reports UIA-style client-update recovery on
  VGG-16/ImageNet with batch sizes `16, 32, 64, 128`.
- The appendix freezes VGG-16 convolutional layers and trains only the fully
  connected layers for this UIA result.
- Reported Table 5 LPIPS values are:
  - GMA: `0.598, 0.613, 0.625, 0.639` for batch `16, 32, 64, 128`.
  - CPA + FIA: `0.410, 0.429, 0.431, 0.448` for batch `16, 32, 64, 128`.

## Local Upstream 4-bit Trend Check

The upstream 4-bit check inserted quantization only in upstream gradient
collection. Upstream ICA, FIA, metrics, and matching were otherwise unchanged.

| row | batch size | quantization | LPIPS |
|---|---:|---|---:|
| upstream CPA/FIA | 16 | 4-bit stochastic per-row | 0.5340 |
| local CPA (미수정) + naive dequant | 8 | 4-bit stochastic per-row | 0.5439 |
| upstream CPA/FIA | 32 | 4-bit stochastic per-row | 0.5615 |

Footnote: the batch sizes differ (`B=8/16/32`), so this validates trend
agreement rather than same-condition equality.

The historical raw method key `naive` includes the ported CPA path. The current
FedAvg naive CPA baseline also includes CPA; the source-separation ablation is
explicitly no-CPA (spec section 0.9). Both no-CPA control conditions were
invalidated for failed variable control on 2026-09-09, so they cannot establish
reconstruction success or CPA's contribution. See the
[invalidation record](NO_CPA_CONTROL_INVALIDATION_20260909.md). The raw B=8
CPA value remains a separate observation. 4-bit GMA remains closed because
fp32 GMA failed to validate its baseline.

## Author Code Protocol

Local source: `C:/Users/조상제/Desktop/CPA/official_cpa/src/collect_updates.py`.

- Dataset/model: `--ds imagenet --model vgg16`.
- Pretrained ImageNet VGG-16 is used.
- For convolutional models, `return_z=True` records FC input embeddings `z`.
- The update is collected by cloning the original model, running local SGD, and
  saving trained-minus-original weights. The code calls this `p0 - p1`, with
  `p0` taken from the trained model and `p1` from the original model.
- Defaults in the author code:
  - `n_rounds=10`
  - `batch_size=32`
  - `n_epochs=10`
  - `lr=1e-3`
  - optimizer: SGD
  - test batches are sampled with `shuffle_test=True`
  - local mini-batches use `shuffle=True`

Local source: `C:/Users/조상제/Desktop/CPA/official_cpa/src/vgg16.py`.

- The modified VGG sets classifier dropout probability to `0`.
- `return_z=True` returns the flattened `avgpool` output before the classifier.

## Our Current UIA Protocol

Local source: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/qacpa/uia.py`.

- Dataset/model: VGG-16/ImageNet with torchvision pretrained weights.
- Conv features are frozen and cached once as `Z=f_conv(X)`.
- FC-only local SGD runs on cached `Z`, so image forwards are not repeated.
- The observed matrix is `(W_initial - W_T) / lr` for `classifier.0.weight`.
  Its exact-arithmetic factorization is `sum_t grad(W_t) = B_T Z`; actual
  fp32 parameter subtraction can differ from the directly accumulated factor.
- Dropout is disabled by using eval-mode classifier execution, matching the
  author code's `p=0` dropout modification.
- The diagnostic protocol uses full-batch local steps with no local mini-batch
  shuffle. At `T=1`, its accumulated-gradient factor equals the GIA/FedSGD
  factor; this does not assert bitwise equality of the transmitted fp32 update.

## Match / Difference Table

| Axis | Published CPA Table 5 | Our UIA diagnostics | Status |
|---|---|---|---|
| Dataset | ImageNet | ImageNet validation cache | match |
| Model | VGG-16 | VGG-16 | match |
| Conv layers | frozen | frozen / cached `Z` | match |
| Updated layers | FC only | FC only | match |
| Observed object | client update | individual client update | match |
| Batch sizes | `16,32,64,128` | `16,32` in T-robustness; `16,32,64` in B_eff validation | partial, extend to `128` before FIA table |
| LR | `1e-3` author default | `1e-3` | match |
| Dropout | author VGG sets `p=0` | eval-mode classifier, no dropout | match in effect |
| Local schedule | `n_epochs=10`, local mini-batch size `32`, shuffle | full-batch local steps `T` | diagnostic difference |
| Image metric | LPIPS after FIA | no FIA in current tasks | pending |

## Round and T=1 Bridge

All GIA reconstruction numbers produced before `m6_round_dependence` use the
ImageNet-pretrained VGG-16 checkpoint directly (`round=0`). The UIA spectrum
diagnostics use FC-only checkpoints. Their `T=1` check reported relative L2
`0.0` for `A`, factorized `Y`, quantized factorized `Y_hat`, and `delta`.
Those diagnostic runners used `Y_factor = B_cumulative @ Z`, not
`Y_delta_over_lr = (W_initial-W_T)/lr`. The factorization supplies the
exact-arithmetic GIA bridge, but the old zero-error report must not be quoted
as a bitwise check of the actual fp32 transmission. The queued baseline below
uses the actual parameter difference and reports its factorization residual.

Threat-model note: the reported leakage assumes a single observed client
transmission. If multiple transmissions from the same client are observable,
the one-transmission values are lower bounds on what temporal access could leak.

## Exact-Published Mode To Use Before LPIPS Comparison

For direct Table 5 LPIPS comparison, run a separate UIA/FIA mode with:

- `n in {16,32,64,128}`
- local mini-batch size `32`
- local epochs `10`
- SGD `lr=1e-3`
- classifier dropout disabled
- conv frozen / cached `Z`
- same FIA path and LPIPS implementation already used in Milestone -1/M3

The current full-batch `T` sweep is the mechanism diagnostic. It establishes
that leakage bias is insensitive to local step count under fixed cached
features; it should not be silently substituted for the exact Table 5 LPIPS
protocol when reporting published numbers.

## Prerecorded next run: UIA fp32 n=16, T=10, CPA Stage 1 + FIA

Protocol audit recorded **2026-09-08 21:00:06 UTC / 2026-09-09 06:00:06 KST**,
before this queued UIA experiment executes. This section supersedes the broader
future grid above for the currently authorized run. The existing M7 controls
must finish first; the queue then runs this fp32 UIA baseline only. Stage 2,
`--refine-A`, GMA, and every 4-bit run remain prohibited. Completion leads to
a report and stop, regardless of the LPIPS outcome.

Run tag: **`m6_uia_fp32_B16_fia`**. The authoritative machine-readable record is
[`m6_uia_fp32_B16_fia.json`](../../quantization_aware_cpa/configs/m6_uia_fp32_B16_fia.json),
locked at `2026-09-08T21:02:30.9834296Z` with SHA-256
`deaf2865732e28d02351b40cd1920f3561b1ea7088805fa00b6c3030fec3ff4e`.
Its registration phase precedes UIA observation collection, ICA, and FIA.

### What Table 5 establishes, and what it leaves unspecified

Appendix B, page 11, reports frozen-convolution VGG-16/ImageNet FedAvg UIA.
Its CPA+FIA values at client sizes 16/32/64/128 are
**0.410/0.429/0.431/0.448**. The n=16 reference is 0.410; 0.448 belongs to
n=128. The paragraph heading says FedSGD while its body and table specify
UIA/FedAvg. The caption mentions a combined attack, but the actual table has
only GMA and CPA+FIA rows. Its paragraph does not state an exact execution
command, seed, evaluated image count, or checkpoint round. Accordingly, the
new run is a documented comparison to Table 5, not a claim of identical
published conditions. [Primary PMLR paper, Appendix B/Table 5](https://proceedings.mlr.press/v202/kariyappa23a/kariyappa23a.pdf#page=11)

The public collection script resets the model from `model_orig` inside every
`n_rounds` iteration. Thus its default ten collection rounds are **ten
independent attacked client updates**, not ten sequential global training
rounds. It defaults to ten local epochs, mini-batch size 32, and SGD lr=0.001.
For n=16, each epoch has one mini-batch, so ten epochs equals ten full-batch
steps; shuffling can change floating-point reduction order but not batch
membership. The stock script's client-size list is 32/64/128/256 and it lacks
a conv-freezing option. The local `official_cpa` copy adds those collection
options to implement the paper's n=16/frozen-conv setting.
[Author collection source](https://github.com/facebookresearch/cocktail_party_attack/blob/main/src/collect_updates.py)

Author `setup()` seeds Python, NumPy, and Torch at 42. This identifies a code
default, not the paper's undisclosed Table 5 seed. The present deterministic
victim seed 0 and reset-per-FIA-call seed 42 are explicit local choices.
[Author setup and seed source](https://github.com/facebookresearch/cocktail_party_attack/blob/main/src/utils.py)

### Exact local protocol, frozen before execution

| Axis | Table 5 / author-code evidence | Queued local value and comparison status |
|---|---|---|
| Attack | CPA + FIA on UIA | CPA Stage 1 followed by unchanged `qacpa.fia.run_direct_fia`; no further stage |
| Precision | Uncompressed reference | fp32 only; no quantizer, no dequantization |
| Dataset and model | ImageNet, VGG-16 | ImageNet `val`, torchvision `IMAGENET1K_V1`, transformed to 224x224 |
| Initial checkpoint | Paper does not identify an attacked global round | Pretrained round 0, restored separately for each update; no M6 trained checkpoint |
| Client training-set size | n=16 Table 5 column | 16 distinct-label images per victim batch |
| Independent attacked updates | Author collection default 10; Table 5 count not stated | Five updates, victim batch indices 0..4; one observed transmission per update |
| Evaluated images | Table 5 count not stated | All 80 originals, global indices 0..79; no retained-only subset and no truth-source supplementation |
| Victim sampling | Author shuffled test sampling; no distinct-label constraint in collection | Existing deterministic seed 0 caches; distinct labels within each 16-image batch; store exact image paths |
| Frozen model portion | Paper freezes convolutional layers | Conv features computed once and cached as `Z`; local SGD updates FC layers only |
| Dropout | Author VGG sets classifier dropout p=0 | Classifier eval mode; same effective absence of dropout |
| Local optimizer | Author SGD lr=0.001 | SGD lr=0.001, momentum=0, weight_decay=0; mean cross-entropy |
| Local schedule | Author epochs=10, mini-batch capacity=32 | T=10 steps over all n=16 samples; matches step count for this n only |
| Attacked matrix | Client first-FC-layer weight change | `(W_initial - W_final) / 0.001`, shape 4096x25088, from actual fp32 parameter subtraction |
| CPA component count | n sources | Full 16 components; no MP truncation, no covariance debiasing, no rank-dependent victim removal |
| CPA optimizer/budget | Existing local CPA implementation | 25,000 Adam steps, lr=0.001, no learning-rate scheduler |
| CPA objective coefficients | Existing ImageNet recipe | decor=5.3, temperature=7.7, nv=0.13, l1=5, tv=0.1, ne=1; implemented NE is zero and embedding TV is inactive when nv>0 |
| CPA initialization | Author/local identity unmixing initialization | Identity matrix; generic Stage-1 seed 0+batch is logged but does not create random restarts |
| Target preparation and matching | CPA absolute-value target convention | `abs(S_hat)` from Stage 1; within-batch permutation-only Hungarian on absolute source cosine; no oracle scaling or `S_true` substitution |
| FIA iterations/optimizer | Existing M3 direct FIA path | 25,000 iterations, Adam lr=0.1, cosine schedule, FI=1, TV=0.1, logging every 5,000 |
| FIA initialization | Table 5 per-call seed schedule unstated | Seed 42 reset for each 16-image invocation; five invocations; existing CuDNN benchmark behavior retained |
| Image metric | LPIPS | Existing TorchMetrics VGG LPIPS, RGB [0,1] mapped to [-1,1]; per-image, per-batch and full-80 means |
| Numerical gates | Local correctness requirements | Observed-vs-factor relative L2 <=0.001; unchanged whitening tolerance 0.001; stop if fewer than 16 sources |
| Predicted and measured counts | Local diagnostic definitions | Predicted: positive top-16 centered observed eigenvalues above zero-noise MP edge=0; measured: matched absolute source cosine >0.9 over all 16 |
| Numerical source rank | Separate local diagnostic | Top-16 singular values `sqrt(25088 * lambda)`, tolerance `max(1e-8,1e-5*s_max)`; never a full-matrix rank gate or source-removal rule |
| Required interpretation | Table 5 comparison | Report exact n=16 gap from 0.410 and local protocol differences; no automatic follow-up experiment |

The exact source update convention deserves emphasis. In author
`collect_updates.py`, the expression named `p0-p1` pairs the trained model
first and the original model second, so it stores **W_final-W_initial**.
The local sign/learning-rate convention above is its globally rescaled
negative. Log this convention explicitly; do not interpret the variable names
as initial-minus-final in the author code.

`qacpa.uia` also exposes `Y_factor = B_cumulative @ Z`. It is a diagnostic
factorization using internal training information. **The new attack must use
`Y_delta_over_lr`, never `Y_factor`**, even if fp32 parameter subtraction
causes a residual. Log the relative L2 and maximum absolute residual between
them. A failed numerical or source-count gate stops the experiment; it cannot
be repaired with oracle factors, exact embeddings, or a reduced denominator.

Local `qacpa.stage1` uses symmetric double-precision covariance/eigendecomposition
and denominator d for whitening, while author whitening uses its original
floating-point eigendecomposition and d-1. The CPA objective is retained, but
these numerical differences and the sampling/FIA seed differences above
prevent calling this byte-identical author-code reproduction. No tuning is
permitted after the baseline result is observed.

### Interpretation boundaries

The user's decision bands are an outcome interpretation rule, not permission
to launch 4-bit work. Apply these rules to the **unrounded full-80 mean**:

| Mean LPIPS | Recorded decision | Action |
|---|---|---|
| 0.410 <= L <= 0.448 | `reference_range_met` | Record reference-range agreement; stop |
| 0.450 <= L <= 0.500 | `protocol_difference_review` | Identify protocol differences and footnote them; retain FedAvg; stop |
| L >= 0.550 | `pipeline_review` | Inspect the pipeline; stop |
| L <0.410, 0.448<L<0.450, or 0.500<L<0.550 | `unclassified_review` | Unspecified outcome; report and review without selecting a new cutoff |

Missing images, nonfinite values, or failed protocol gates are failures to
complete a valid baseline, not entries in a favorable LPIPS band. Every branch
ends with reporting and stopping. The wider 0.410-0.448 band spans different
published client sizes and does not make an n=16 result of 0.448 an exact
replication of the published n=16 value, 0.410. No branch authorizes 4-bit.

The old fp32 0.4448 is a B=8 GIA reference combining 36 CPA images with four
exact-source FIA supplements. It is neither this n=16/T=10 UIA baseline nor
evidence of Table 5 replication. Its companion 4-bit 0.5439 remains a separate
historical M7 control comparison.
