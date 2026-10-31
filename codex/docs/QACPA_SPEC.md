# Quantization-Aware Cocktail Party Attack — Implementation Spec (v3)

> Paste into Codex. English is deliberate: this is full of tensor shapes and
> identifiers, and ambiguity costs days.
>
> **v3 supersedes v1/v2.** It reflects the first Milestone-1 run, which found a
> Stage-1 whitening bug and an unsettled Milestone -1.
>
> **Read §0 and §1 before writing any code.**

---

## 0.17 Authorized uniform random filling, Top-k10 only (2026-09-12)

The user now authorizes the same sparse experiment with missing coordinates
filled from U(-tau,tau), tau=min(abs(retained packet values)),20 candidates
per original batch, FIA5k screening, and the top3 continuing to total25k.
The follow-up explicitly selects **10% only**. Reuse the same80 images,
original five n16/T10 round0 FC-only SGD updates, and byte-identical existing
Top-k10 packets. Retained FP32 values are unchanged; random filling is
attacker-local postprocessing with no added communication.

Original CPA/FIA objectives, hyperparameters, seeds and post-FIA matching
remain unchanged. The new staged FIA driver fixes cosine horizon25k from
iteration1 and restores Adam/scheduler/pixels/RNG at5k. Select top3, then one
whole16-image final candidate per batch, by observable recompressed-update
relative L2 with predicted labels from the public round0 model. No true
images/labels/features or missing true values enter filling or selection.
Evaluation follows final selection, retains all80, and reports per-source
cosines and LPIPS/SSIM/PSNR. Save all20 screens and3 final candidates.

This user-requested search budget supersedes the equal-budget rule only for
this authorized arm: per batch CPA500k and FIA160k, versus one CPA25k/FIA25k
zero-fill baseline. Do not describe the comparison as isolating filling
from additional compute, and do not launch extra control arms. No Stage2,
debias, MP, LS_A or oracle substitution. See
`quantization_aware_cpa/docs/RANDOM_FILL_SAME80.md` for the frozen protocol.

## 0.16 Authorized Top-k 30/20/10% on the same80 (2026-09-11)

The user explicitly requests preserving the completed Top-k50 method while
testing retained fractions30%,20%,10%. This supersedes section0.15's ratio
restriction for these three conditions. Reuse the identical80 images and five
cached actual n16/T10 round0 FC-only updates. Only the retained count changes:
floor(N * percent / 100), global magnitude within FC1, ascending flat-index
ties, unchanged FP32 values plus bitmap, receiver zero fill. Preserve all
CPA/FIA settings, seeds, source adapter and post-FIA evaluation. Execute the
three conditions sequentially on CUDA, with separate frozen configs/outputs;
retain existing50% and packed4 artifacts unchanged. No training, CoRA,
additional ratios or tuning. Report all80 for every condition and actual FC1
packet bytes. See the [30/20/10 protocol](../../quantization_aware_cpa/docs/TOPK30_20_10_SAME80.md).

## 0.15 Authorized Top-k 50% sparsification on the same80 (2026-09-11)

The user explicitly requests the same pipeline with only communication
compression replaced by magnitude Top-k 50% sparsification. This authorizes
implementation and execution of `cpa_fia_topk50_same80_B16`, preserving all80
images, the original five n16/T10 cached actual updates, and the completed
same80 packed4 run's upstream CPA/FIA settings, seeds, abs adapter and
post-FIA evaluation. No new training, new attack arms, or tuning is authorized.

Apply global magnitude selection within the attacked classifier.0.weight
tensor, retaining exactly half its coordinates with ascending flat-index
tie breaking. Serialize original FP32 retained values plus a position bitmap;
decode from this actual packet with zero fill. No quantization, rescaling,
error feedback, or desparsification prior. Keep the existing experiments and
their frozen artifacts unchanged. Report all80 image/source metrics, per-batch
values, native reconstruction PNGs, actual packet bytes and provenance.
See [Top-k 50% protocol and CMD usage](../../quantization_aware_cpa/docs/TOPK50_SAME80.md).
Stop after this requested condition; no other sparsity ratios are in scope.

## 0.14 Reuse the original historical80 for packed4 CPA+FIA (2026-09-10)

The user now requests the same 80 images used before. Prepare a separate
`m10_cpa_fia_packed4_same80_B16` entry point with the original five n16/T10
batches, all global IDs 0-79 in original order, and unchanged M6 input tensors
and actual saved updates. This supersedes the best10 sample restriction for
the new condition only. The batch grouping uses the existing16x5 default
communicated in this session; do not silently regroup into one n80 update.
No LPIPS selection or retraining is needed with the original grouping.

Keep the real packed4 codec, upstream CPA/FIA settings and post-FIA evaluation.
Only the packed4 CPA+FIA condition is in scope. Preserve historical best10 files
and results. The setup task verifies the source hashes and prepares five real
packets on CPU; GPU reconstruction is a separate `--phase run` command.
Do not label preparation or packet creation as completed reconstruction.
See [same80 usage](../../quantization_aware_cpa/docs/M10_SAME80_PACKED4.md).

## 0.13 Execute only packed4 CPA+FIA on selected best10 (2026-09-10)

The user requested only the CPA+FIA 4-bit arm, then restricted the client data
to the best10 of the stored80 images. This supersedes section0.12's four-arm
execution design, fixed n16 and no-new-training constraint for this one run.
Select historical M8 LPIPS ascending (global_index tie-break), first10, before
new results: IDs `[46,8,58,39,1,17,63,9,53,70]`. Keep their cached images/features
and labels, and form one new n10/T10 round0 FC-only SGD0.001 actual update using
the unchanged existing collector. Other training settings remain fixed.

Run tag `m10_cpa_fia_packed4_best10`; config with the same name under configs.
Use the real packed4 codec defined in section0.12, unchanged upstream CPA/FIA
algorithms and25000-step budgets, full10 sources, abs adapter and post-FIA
evaluation matching. Execute no FP32 attack, raw-row attack, Stage2 or other
grid. Historical n16 scores select the images only; they are not a paired
baseline for this newly formed n10 batch. Label this as a historical best10
selected exploratory sample. Report every one of the selected10 images,
packet bytes, image/source metrics, input/config/code fingerprints and runtime.
Stop after this single requested condition.

## 0.12 Current communication compression redesign (2026-09-10)

The user's current request is to inspect recent quantization and redesign the
experiment with only compression changed. The completed design is
[COMMUNICATION_COMPRESSION_REDESIGN_20260910.md](COMMUNICATION_COMPRESSION_REDESIGN_20260910.md),
with `quantization_aware_cpa/configs/m10_compression_only_B16.protocol.json`.
It specifies actual per-row stochastic 16-level integer codes, two codes per
byte, FP32 radii and a decoder that receives only the packet. Existing M8/M9
attacks, cached actual n16/T10 updates, all80 images and budgets remain fixed.
Fresh FP32 arms use each same attack; historical M6 is not the M8 FP32 baseline.

This is a design deliverable; the new codec and attacks have not been run.
It supersedes the historical quantizer-preservation rule for this redesign,
without relabeling old continuous-dither results as integer-codec evidence.
Actual stochastic rounding does not satisfy the old half-delta cell or
delta-squared/12 noise model; M8/M9 already omit those corrections. Compression
bytes are scoped to the attacked FC1 weight packet. Historical results and
protocol locks remain unchanged; no new success or utility result is claimed.

## 0.11 M9 authorized FedAvg raw-row FIA baseline (2026-09-10)

User authorization "ㅇㅋ ㄱㄱㄱ" permits the new
`m9_fedavg_rawrow_dither4_B16` experiment under the
[fixed raw-row protocol](FEDAVG_RAWROW_FIA_PROTOCOL.md). Read the exact five
saved M8 dither4 observations, select16 raw rows per batch with CPU seed
`0+batch` and `randperm(4096)[:16]`, keep zero rows, and pass their absolute
values to the original Direct FIA. Freeze all row IDs and input/code/config
fingerprints before FIA. No normalization, whitening, CPA, LS-A, debias, MP,
Stage2 or new client training. Use the same FIA budget as M8 and only after
FIA apply M8's signed-source absolute-cosine Hungarian permutation for all80
metrics. This is raw rows + abs + FIA, not dequantization alone.

This authorization does not restore the invalid M7 no-CPA results or any
dependent threshold, normalization or contribution claim. It authorizes only
this declared row-adapter baseline, not a claim of a unique fully controlled
ablation of every part of CPA. See STATUS for preparation/execution state.

## 0.10 No-CPA control invalidation (2026-09-09; current reporting authority)

The user has rejected `m7_no_cpa_control` (both fp32 and 4-bit) because
experimental variables were not adequately controlled. These results are
**scientifically invalid**, regardless of completed execution records.
This section overrides all earlier instructions that treat this control as
valid, calibrate thresholds from it, normalize against it, or infer CPA's
contribution from it. No specific confound is asserted beyond the user's statement.

Withdraw the no-CPA contrasts and combined IUT claim, all no-CPA-referenced
normalization, both historical rank-based and later common-cutoff success
counts, and dependent figures. Preserve raw data and locked protocols as
provenance. Do not reuse these outputs as evidence or invent a replacement
threshold. See [the invalidation record](NO_CPA_CONTROL_INVALIDATION_20260909.md)
and `quantization_aware_cpa/results/result_validity.json` for dependency scope.

The current requested figure contains only the completed same80 client-update
4-bit experiment and its paired fp32 reference: `m7_uia_4bit_B16` and
`m6_uia_fp32_B16_fia`. LPIPS, PSNR, SSIM and measured embedding cosine do not
use the invalid control. Section 0.9's direct upstream FedAvg naive CPA run
and independent reconstruction data remain outside this invalidation.

## 0.9 Direct upstream FedAvg naive CPA baseline

The latest user instruction authorizes `m8_fedavg_naive_cpa_dither4_B16`:
direct original CocktailPartyAttack + Direct FIA on the same saved n16/T10
observations. [Fixed protocol](FEDAVG_NAIVE_CPA_PROTOCOL.md). No LS-A calculation,
debias, MP, Stage2, refinement or new client training. The input remains the
existing dither4 model and must not be called an integer4bit codec.

This amendment also supersedes the earlier rule reserving `naive` for no-CPA.
**FedAvg naive CPA includes CPA**; **no-CPA** explicitly omits it. Historical
raw keys/numbers are preserved. Earlier `CPA (미수정)` wording means the old
unrefined CPA-based port; it is not a verified unchanged-upstream-code claim.

## 0.8 M7 follow-up: confirmed corrections and authorized Tasks 1-5 (2026-09-09)

This user-authorized amendment supersedes earlier stop/4-bit restrictions for
`m7_uia_4bit_B16` only. Stage 2, `--refine-A`, new model training, GMA4bit,
seen-batch round dependence, n degradation curves, extreme non-IID and a new
(B,bit) grid remain out of scope. Reuse stored client updates; do not train again.

Confirmed reporting values:
- Prediction validation: **81/96 exact (MAE 0.219)**, one authoritative overall count.
- UIA fp32 n=16/T=10, all80: **LPIPS 0.4546 / PSNR 22.406 / SSIM 0.3427**;
  matched embedding |cos|>0.9 is79/80, no oracle supplementation.
  CPA Table5 CPA+FIA spans 0.410-0.448 over batch 16-128. The current 0.4546 is
  **0.0066 above the upper endpoint**, a range-boundary comparison across n;
  it is not inside the interval or an identical-condition reproduction claim.
- Canonical GIA fp32 reference: **0.4465, n=36**. Recomputed after excluding
  the four oracle supplements (global22,24,32,36); the old fp32 n=36 issue is closed.
- **WITHDRAWN:** similarity of no-CPA means does not establish a valid control.
  Both no-CPA conditions are invalid under section 0.10.
- UIA queue failed versus completed outputs is a suspected exit-code handling bug;
  keep original queue logs and record the discrepancy in STATUS.md.

The exact follow-up analysis rule is [M7_FOLLOWUP_ANALYSIS_PROTOCOL.json](M7_FOLLOWUP_ANALYSIS_PROTOCOL.json),
with an adjacent timestamp/hash lock. Historical results were already available:
this is a transparently dated follow-up amendment, not a retroactive preregistration.
The original locked M7 protocol/results remain archived and unchanged.

1. `m7_normalized_metrics`: **WITHDRAWN**. All no-CPA-referenced distances and
   retained ratios are invalid; the locked formula is retained only as history.
2. `m7_shuffle_control`: seeds 0-19, independent whole-set permutations for
   fp32pure36/4bit40/UIAfp3280; retain/log fixed points, never rematch.
   Save per-pair rows,20 seed means, mean/sample SD(ddof1)/linear5th and95th
   percentiles for both seed means and pooled per-pair distances.
3. `m7_perimage_distribution`: **PARTIALLY WITHDRAWN**. The no-CPA-derived
   cutoff and every success count or interpretation using it are invalid.
   Independent CPA/UIA LPIPS observations and descriptive distribution shape
   are preserved; they do not supply a replacement success threshold.
4. `m7_uia_4bit_B16`: same80 stored actual UIA updates, n16/T10, pretrained
   VGG16/frozen conv,4bit stochastic per-row; full16 CPA without attack debias/MP
   truncation, no oracle fills. Existing MP B_eff is a separate diagnostic.
   FIA25000/TV0.1/Adam lr0.1/cosine/seed42 per16, matching fp32 exactly.
   Record all requested metrics,runtime and random12 seed42 not-cherry-picked grid.
5. `m7_consistency_audit`: recompute pure36 fp32 loss-LPIPS Spearman; align
   prediction vs prediction by all condition keys; recount96 and full T1 subset,
   separately identify the core per-row subset; square scatter uses separate
   T=1 (= FedSGD) and T=10 (= FedAvg) markers.

Optional TV/iteration and granularity-error investigations follow Task4 only.
The79/80 cosine count by itself does not prove the0.0066 gap is caused by FIA.

Implementation scope: repository `qacpa.quantizer.quantize(rounding='stochastic')`
models subtractive-dither observations as `Y_hat=Y+U(-delta/2,delta/2)`, with
`delta=2*row_max_abs/(2**bits-1)` and its existing numerical cell projection.
The authorized run retains this existing observation model. It does not test
an integer transmission codec; conclusions about deployed 4-bit codecs need
that distinction. The per-row seed is `0+1000003*batch_index`.

<!-- M7_FOLLOWUP_RESULTS_BEGIN -->
### Section 0.8 execution results and validity correction

Execution completed, including the same80 4-bit UIA run. Completion does not
validate the no-CPA control. The previous normalized retention, common LPIPS
cutoff, derived counts and majority-success interpretation are withdrawn under
section 0.10. Independent source/image measurements are preserved.
Canonical fp32 rho=+0.031 versus 4bit -0.591; 81/96 exact (MAE0.219), allT1=39/48.
Per-row-only T1=30/36 is a separate subset. The n32 difference is five-batch
mean21.2 versus matched batch0 prediction24; quantizer seeds differ.
See [the current review](EXPERIMENT_RESULTS_20260909.md) for sources.
<!-- M7_FOLLOWUP_RESULTS_END -->

## 0.7 historical authorized successor -- UIA fp32 n=16 FIA (2026-09-09)

After the currently running M7 control pipeline completes and exits, execute
exactly one UIA condition: `run_tag="m6_uia_fp32_B16_fia"`, uncompressed fp32,
VGG-16 pretrained round-0 checkpoint, frozen convolutional features, FC-only
SGD lr=0.001, 10 local steps, n=16. Five independent client updates reset the
model to round 0; these are five measurements, not five sequential server rounds.
Use all 80 images from victim seed0 batches0-4. Log per-row scale granularity
as not applicable to fp32. Do not invoke the old M6 GMA-first launcher.

The full pre-execution comparison with the paper and author code is in
[`UIA_PUBLISHED_PROTOCOL.md`](UIA_PUBLISHED_PROTOCOL.md). The frozen machine
config is `quantization_aware_cpa/configs/m6_uia_fp32_B16_fia.json` with its
adjacent `.lock.json`; SHA-256 is
`deaf2865732e28d02351b40cd1920f3561b1ea7088805fa00b6c3030fec3ff4e`.
The paper reports n=16 LPIPS 0.410; 0.448 belongs to n=128. Do not describe
the whole 0.410-0.448 cross-n range as identical-condition reproduction.

Feed the actual observed `(W_initial-W_after_T)/lr` into unchanged Stage1
CPA. `B_cumulative @ Z` is an oracle factorization diagnostic, never a
replacement attack input. Preserve full16 components with no MP truncation or
debiasing, the existing CPA objective/budget, and the existing whitening gate.
Abort on invalid factorization (relative L2>0.001) or whitening; do not fill
missing targets with exact source embeddings, reduce the denominator, or adjust
the rank to obtain better LPIPS. FIA uses `abs(S_hat)` and permutation-only
source matching, 25000 iterations, TV=0.1, FI=1, lr=0.1, Adam, cosine, seed42
per16-image invocation, exactly the existing GIA FIA settings.

Report all80 per-image LPIPS/PSNR/SSIM and source cosines, means, distributions,
per-victim-batch results and runtime by stage. At zero quantization noise, count
positive top16 centered observed covariance eigenvalues for predicted B_eff;
report the nominal no-noise ceiling16 and a separate top16 numerical-rank
diagnostic. Measured B_eff counts matched source |cosine|>0.9, without selecting
the FIA image set. Undefined eigenvalue/zero-MP-edge ratios are null, not finite
noise margins. Document float32 update-rounding residuals rather than claiming
the old ideal-factor T1 bridge is bitwise exact for transmitted updates.

Apply the user's bands to the unrounded full80 mean LPIPS: [0.41,0.448]
records reference-range agreement; [0.45,0.50] calls for protocol attribution
and a footnote while retaining FedAvg; >=0.55 requires pipeline review.
Values in unspecified gaps or below0.41 are unclassified and require review.
**Every outcome stops at this fp32 result.** Stage2, `--refine-A`, 4-bit and
GMA remain prohibited; a favorable fp32 number does not authorize automatic
4-bit execution in this queue.

The successor waits for the existing M7 OS process to exit and requires both
its execution and report completion markers, with the registered protocol hash.
It does not poll reconstruction metrics, modify M7, or start concurrent GPU
work. If M7 fails, the successor records that dependency failure and stops.

## 0.6 v14 historical locked task -- M7 no-CPA controls (2026-09-09)

**HISTORICAL, NOT AN ACTIVE ANALYSIS RULE:** this section records the original
protocol. Section 0.10 invalidates the no-CPA control and every dependent
claim or success rule below. Locked protocol files remain unchanged.

This explicitly authorized task supersedes the older reconstruction restrictions
only for the following controls. Stage 2, `--refine-A`, Stage 3 and 4-bit GMA
remain closed. The superseded M6 queue with `-RunGma4Bit` is stopped.

The authoritative preregistration is
[`M7_CONTROL_PREREGISTRATION.json`](M7_CONTROL_PREREGISTRATION.json).
Freeze its SHA-256 and timestamp before computing any new M7 LPIPS. Never
change that protocol after inspecting the controls; amendments require a new
version and must be identified as amendments. Store numerical thresholds before
counting candidates, using the formula below, without selecting a cutoff from
the CPA results.

Fill `{fp32,4-bit} x {CPA present,CPA absent}` for B=8, seed=0, the same original
40 images (batches 0-4). Reuse fp32 0.4448155567 and 4-bit 0.5439403489.
**The reused fp32 row is a historical hybrid: 36 CPA images plus exact-source
FIA at global indices [22,24,32,36].** Disclose this in the 2x2 table and show
a same-36-image sensitivity; do not call the full fp32 row a pure causal CPA
comparison. The 4-bit row is all 40 historical CPA reconstructions.

`naive` now means only the **no-CPA control**. Historical raw method keys,
paths and immutable metrics called `naive` remain provenance identifiers; their
display label is **CPA (미수정) + naive dequant**. They always include CPA.

For `m7_no_cpa_control`, select eight observation rows per batch with
`torch.randperm(4096, generator=CPU_generator.manual_seed(0+batch_index))[:8]`.
Use exactly those indices in fp32 cached `Y` and legacy 4-bit cached `Y_hat`.
All selected rows, including zero rows, are retained. The sole transform is
row L2 normalization with epsilon 1e-12: no centering, whitening, projection,
ICA, unmixing, sign flip, absolute value, clipping, or row-quality selection.
The observation has 4096 rows, so this fixed random subset provides the required
eight FIA targets without an oracle choice among rows.

Use unmodified `qacpa.fia.run_direct_fia`: 25000 iterations, FI=1, TV=0.1,
lr=0.1, Adam, cosine scheduler and FIA seed=42 per invocation. Match historical
4-bit invocation grouping [0:16], [16:24], [24:32], [32:40]. The victim B stays
8; the first FIA call processes two victim batches. Historical fp32 invocation
sizes differ because of missing sources; disclose that initialization limitation.

For `m7_prior_only`, generate one iid Uniform[0,1) CPU tensor (40,25088) with
embedding seed=42 and normalize its rows. It uses no observation or true
embedding. Invert with exactly the same FIA settings and grouping. Use one
shared prior-only control for both precisions. This isolates random-target FIA
and TV together; it is not a claim about TV alone.

Ground truth is used only for evaluation. After reconstructing fixed targets,
use the existing within-batch Hungarian assignment on absolute target cosine
for permutation-only evaluation, without applying its sign or scale correction
to any no-CPA target. Also retain positional LPIPS as a matching sensitivity.
LPIPS remains the existing VGG metric and RGB[-1,1] convention.

For `m7_recompute_controls`, reuse float reconstructions, verify recalculated
historical diagonals within 1e-5, and compute shuffled pairings under NumPy seeds
0-19 separately for fp32 and 4-bit. Ordinary permutations may retain fixed
points: log them and never rematch after shuffling. Constant controls are the
pixelwise mean of the complete transformed ImageNet val split (log count), gray
0.5, and iid Uniform[0,1) RGB noise with torch seed=42. Do not substitute the
40 evaluation-image mean for the dataset mean. These are empirical reference
levels, not a mathematical LPIPS floor (identical images define zero).

**Success rule, fixed before new results:** for an image from victim batch b,
exclude all eight images of b from each reference. For each remaining 32-image
precision-matched no-CPA and prior-only reference compute
`p=(1+count(L_control <= L_candidate))/33`. Count a control-calibrated success
iff both p-values are <=0.05; equivalently the image LPIPS must be strictly
below both reference minima. This conservative rule has minimum p=1/33.
Compute and save the per-batch control-only thresholds before applying them to
the candidates. Counts are descriptive lower-tail exceedances, not individual
significance or familywise-error-controlled discoveries: one control realization
per image does not support such a claim.

The confirmatory claim is 4-bit CPA better than **both** no-CPA and prior-only.
Average paired control-minus-CPA differences within each of five victim batches;
enumerate all 32 batch sign flips for a one-sided mean-difference test (include
ties). The intersection-union p is the maximum of the two p-values. Declare
group evidence only when both effects are positive and p<=0.05. The minimum
possible p is 1/32, so sample size and sign-symmetry assumptions must be stated.
The fp32 hybrid and same-36 sensitivity are secondary. Shuffled, constants and
image success counts cannot independently establish the CPA contribution claim.

Deliver a 2x2 LPIPS/count table, per-image CSVs, one common-axis histogram
overlay including both CPA rows, both no-CPA rows, both shuffled rows,
prior-only and each constant, and a no-CPA image grid sampled with NumPy seed42
(12 of 40, without replacement) labelled **not cherry-picked**. Persist seeds,
row indices, source cosines per image, input hashes, full config, per-batch
metrics, wall time and run tags. No Stage-1/Stage-2 diagnostics are computed for
a pipeline that explicitly omits those stages; record them as not applicable.

## 0. v4 update -- CPA ICA replacement

Milestone -1 is complete: upstream CPA was ported to the local Python/Torch
stack and reproduced at fp32, `B=32` (LPIPS `0.466` vs paper `0.483`).
The upstream unmixing objective, as implemented in code, is not FastICA and does
not use the paper-described neg-entropy term in practice: the computed
neg-entropy contribution is overwritten to zero. The active objective is
decorrelation + non-negativity penalty + L1 + optional TV, optimized over a
row-normalized unmixing matrix with Adam/SGD.

Stage 1 keeps the quantization-aware v3 front end: diagonal covariance debiasing,
MP threshold rank truncation before whitening, and `B_eff` metrics/source
matching in the retained subspace. Only the ICA backend is replaced:
`sklearn.FastICA` is no longer the default path. The default backend is the
upstream CPA-style soft-penalty unmixing optimizer. The CPA ICA L1 coefficient
is logged separately from Stage 2 `lam_R` because both priors can be active.

The required v4 sweep must be interpreted independently from the old FastICA
sweep. Under CPA ICA, `c_mp=1.0` no longer collapses to `0/8`; however, it still
keeps weak near-noise axes. The observed 4-bit plateau remains around
`c_mp in {3, 6}`, while `c_mp=12` over-truncates one batch.

## 0.1 v5 correction -- LS-scaled A_hat for cell constraints

CPA ICA's row-normalized unmixing matrix is appropriate for recovering
`S_hat`, but it discards absolute mixing scale. Stage 2 cell constraints depend
on absolute scale, so after ICA the active `A_hat` definition is the observed
least-squares reconstruction

```
A_hat = Y_hat @ S_hat.T @ inv(S_hat @ S_hat.T)
```

with a pseudoinverse fallback if the small Gram matrix is singular. This uses
only observed `Y_hat` and recovered `S_hat`, so it does not use oracle data. The
older lifted `P.T @ sqrt(lam) @ A_hat_red` reconstruction is retained only as a
diagnostic (`pre_ls_reconstruction_rel_l2`).

`reconstruction_rel_l2` is not a valid scale check for Stage 2. In the v5
failure case it stayed in the normal `0.47-0.56` band while the dimensionless
cell hinge for `A_hat` was about `4e5`. Frobenius reconstruction error and the
max-norm quantization cell constraints test different geometry. For Stage 2
scale validation, use dimensionless `H`, consistency rate, and residual hinge.

Stage 2 also has an input guard: if a final column has larger dimensionless
cell hinge `H` than the input `S_init`, that column is reverted to `S_init`.
This guard is oracle-free and prevents the refinement from making a column
worse.

Main-claim corrections:

- The "one noisy component contaminates everything" mechanism is a FastICA
  symmetric-orthogonalization observation, not the main CPA-ICA mechanism. Keep
  it only as an appendix/diagnostic note with the deflation run.
- Delete the claim that full-B `bplus` collapses. It reproduced under FastICA,
  but not under CPA ICA.
- The reason for MP truncation is now the Stage 2 constraint system: dropping
  near-noise directions improves consistency and avoids asking Stage 2 to solve
  constraints built from unreliable axes.

## 0.2 v5 Milestone 3 and rank split notes

Milestone 3 feeds recovered source embeddings directly into the upstream-style
Direct FIA image inversion path. The required v5 report artifacts are
per-image LPIPS/PSNR/SSIM CSVs, a JSON summary, PNG dumps, and two 4-column
figures: `original | fp32 | 4-bit naive | 4-bit A_oracle`, once in source order
and once sorted by `per_sample_loss`. Log
`Spearman(per_sample_loss, per_image_lpips)` for each method.

Rank split is a separate post-M3 diagnostic path. The oracle-extra diagnostic
uses `[A_hat_keep | A_true_discarded]` only to test whether the lost
sub-threshold sources are missing because the MP-truncated `A_hat` lacks those
columns. If this succeeds, the non-oracle implementation may estimate extra
columns from `R = Y_hat - A_hat_keep @ S_hat_keep` using a short residual ICA,
then run full-`B` Stage 2 with small update scale on retained rows and full
update scale on extra rows. Do not treat the oracle-extra diagnostic as a
method row.

Milestone 3 image metrics must be compared on a common sample set. The active
full-`B` image row for `A` is now a fallback composition: use Stage-2 `A`
sources for MP-retained samples and the `naive` source for MP-discarded samples.
This makes `A` report `n=B` per batch and prevents a selected-retained subset
from being compared against full-`B` fp32/naive/oracle rows. Always report the
selection-effect check: recompute fp32 and naive on the exact retained global
indices used by the non-fallback `A` row before interpreting
`Spearman(per_sample_loss, per_image_lpips)`.

Record the post-M3 observation that `A_oracle` can be worse than `A` in image
LPIPS even when its embedding cosine is higher. Weak MP-subthreshold components
partially recovered only to cosine around `0.5-0.8` can behave like noisy FIA
targets rather than useful targets. The oracle-extra rank-split diagnostic
showed the same pattern: it recovered more source embeddings, but improved mean
LPIPS by only about `0.0012`. Treat this as evidence that the MP edge is not
only a recoverability boundary for source estimation; it is also an empirical
utility boundary for whether a recovered source should be passed to FIA.

The v8 retained-source diagnostic separates this from the discarded-column
question. On the 30 MP-retained sources, `A_oracle` has higher source cosine
than estimated-`A` on 28/30 sources, with mean retained cosine
`0.999036` vs `0.996184`. On the same retained 30 images, `A_oracle` also has
lower LPIPS (`0.4511` vs `0.4596`). This means the binary `cos > 0.9` source
count hides an `A_hat` precision loss inside the retained subspace.

A first `--refine-A` diagnostic was run as alternating S-step plus one
dimensionless hinge A-step. It is not adopted: every A-step reduced the cell
hinge objective, but the retained source cosine worsened on 30/30 sources
(`0.996184` -> `0.982803`). Hinge-only A refinement is therefore
underconstrained by the quantization cell and should remain a negative
diagnostic unless a new regularized A prior is specified.

The Stage-2 image-gain path is now closed. The decisive evidence is that
`--refine-A` reduced the hinge in 25/25 accepted A-steps while worsening every
retained source cosine. The quantization cell constraint is an inequality, so
the low-hinge A set is broad and is not anchored to `A_true`; with `S_hat`
fixed, hinge minimization lets `A_hat` absorb source error. The oracle-extra
image gain (`~0.0012`), the cases where `A_oracle` is not useful despite higher
source cosine, and the negative refine-A result all indicate cell consistency
saturation under the current Stage-2 formulation.

This is a paper-level negative result, not an implementation TODO. Four
diagnostics close the image-improvement path: dimensionless L1 prior recovers
only `2.7%` of the naive-fp32 LPIPS gap for actual `A`; oracle-extra rank split
adds only about `0.0012` LPIPS; hinge-only `--refine-A` improves H in 25/25
steps but worsens retained cosine in 30/30 sources; debias fallback explains
only `9.0%` of the `A_oracle-A` gap. The mechanism is that retained sources are
already near-perfect under naive (`cosine 0.996-0.999` in four of five
batches), so there is little room for Stage 2 to improve images. Batch 2 is the
exception (`retained naive cosine mean 0.856`, min `0.011`) and is the only
batch with a strong retained LPIPS gain (`+0.0443`).

The remaining `A_oracle` image recovery must be interpreted as an accounting
effect before making method claims. In the full-`n=40` M3 table, `A_oracle`
recovers about 50.4% of the naive-fp32 LPIPS gap, while estimated-`A` recovers
about 2.7%. The `A_oracle - A` mean LPIPS gap is `0.0473`; only `0.0063` of
that full mean comes from the 30 MP-retained images, while `0.0410` comes from
the 10 fallback images where `A` deliberately uses the naive target. Therefore
the large full-row oracle advantage is mostly a fallback-boundary /
set-definition effect, not a reason to continue Stage-2 image tuning.

The v10 debias-fallback check tested the last plausible Stage-2 salvage path:
replace the MP-discarded `A` fallback from no-debias `naive` to full-`B`
debiased Stage-1 sources, without MP truncation. This helps only slightly.
Full-`n=40` LPIPS changes from `A=0.5413` to `A_debias_fallback=0.5370`
(recovery `2.7%` -> `7.0%`), while `A_oracle` remains `0.4940`
(recovery `50.4%`). On the 10 fallback images alone, `naive_full` gives
LPIPS `0.7865` and source cosine `0.2098`; `debias_full` gives LPIPS `0.7693`
and source cosine `0.2935`. Therefore naive fallback is a small part of the
gap, not the missing mechanism.

The oracle-extra vs `A_oracle` diagnostic also refutes the hypothesis that
oracle-extra failed because its weak-source `S` estimates were much worse. For
the 10 sub-threshold sources, oracle-extra final source cosine differs from
`A_oracle` by only `-0.0017` on average. The large apparent mismatch between
oracle-extra image gain (`~0.0012`) and fallback contribution (`~0.0410`) is
therefore mostly selection/threshold/accounting, not an extra-column source
quality gap. Batch 2 is the one positive retained batch because its retained
`naive` source cosine is much lower (`0.8563` mean, one source near `0.0109`);
other batches have retained naive means around `0.996-0.999`.

## 0.3 v11 pivot -- UIA / FedAvg spectrum before any new reconstruction

Milestone-4 GIA Phase A is stopped. The reduced GIA grid and the published-range
GIA table (`B=32-1024`) remain unfinished, but the `B_eff` prediction idea is
not discarded. It moves to UIA/FedAvg, because the attacked observation
`Delta theta / lr` has the same low-rank matrix form as GIA:

```
Delta W_0 / lr = sum_t grad_t(W_0) = B_T Z
```

where `Z = f_conv(X)` is the frozen VGG-16 convolutional embedding and `B_T`
collects the accumulated backprop deltas for the first FC layer. For UIA
diagnostics, compute `Z` once and cache it; local training updates only the VGG
classifier on cached `Z`. Do not run image forwards inside local SGD steps.

Stage 1 of the UIA pivot is spectrum-only: no ICA, no Stage-2 solver, no FIA,
and no `--refine-A`. Run VGG-16 with frozen convolutional layers and FC-only
FedAvg in eval mode so that `T=1` is numerically identical to the existing GIA
single-gradient observation. The required diagnostic grid is
`n in {16, 32}`, `T in {1, 2, 5, 10}`, 4-bit stochastic per-row quantization,
`run_tag="m5_uia_spectrum"`.

For each UIA Stage-1 cell, log true numerical rank from singular values of
`B_T Z`; if it exceeds `n`, stop immediately because the FedAvg setup is wrong.
Also log `||b_i||` distribution and CV next to same-batch GIA `||a_i||` CV, all
top-`n` debiased eigenvalue ratios `lam_deb / mp_edge`, predicted `B_eff(T)`
using the same MP calculation as `v10_mp_prediction_grid`, and
`Spearman(final loss, ||b_i||)`. The `T=1` sanity check must report relative
error against GIA `A_true`, GIA `Y`, and stochastic quantized GIA `Y_hat` under
the same quantization seed.

Only if Stage 1 passes should UIA run the A1-style validation grid with ICA and
Hungarian matching. That later stage is `n in {16,32,64}` by
`bits in {2,3,4,8}` by `T in {1,10}`, plus 4-bit per-tensor/per-bucket
granularity checks for `n in {16,32}`. Stage 2, FIA, `--refine-A`, and the old
GIA grid remain forbidden until explicitly re-approved.

## 0.4 v12 direction -- UIA is the main setting

UIA/FedAvg is promoted to the main experimental setting. GIA is now described
as the `T=1` special case of the same observation model, not as the primary
protocol. This resolves the old GIA published-range issue for the next
comparison step: CPA Appendix B Table 5 reports UIA-style VGG-16/ImageNet
batch sizes in the `16-128` range, and the active `n in {16,32,64,128}` UIA
settings sit inside that published range. Because the attacker observes an
individual client update, secure aggregation is not part of this threat model;
the protocol is still FedAvg, which is the more realistic deployment setting.

The first UIA spectrum run falsified the sample-uniformity hypothesis on one
batch. Leakage bias is insensitive to the number of local steps over
`T in {1,2,5,10}`: for `n=16`, CV changes only from about `1.13` to `1.14` and
`Spearman(final loss, ||b_i||)` stays around `0.93-0.94`; for `n=32`, CV stays
around `1.00-1.02` and Spearman remains `0.93-0.96`. The working mechanism is
that the sample order of `||delta_i||` is preserved across local steps, and
with frozen convolutional features the local classifier update does not
re-order sample difficulty. Therefore increasing local step count is not a
defense unless later multi-batch/seed checks contradict this.

The required robustness check was run over five victim batches and three seeds
with `T in {1,2,5,10,20,50}`. No rank exceeded `n`; observed ranks were
`14-16` for `n=16` and `29-32` for `n=32`. The bias did not wash out with T:
mean CV stayed around `1.14-1.20`, and mean Spearman stayed high
(`n=16`: `0.942 -> 0.905`, `n=32`: `0.967 -> 0.941` from `T=1` to `T=50`).
Predicted `B_eff` rose only mildly (`10.33 -> 11.47` for `n=16`,
`19.27 -> 21.33` for `n=32`). Therefore local-step count alone is not a
privacy defense under frozen-conv UIA/FedAvg.

Required robustness check: rerun spectrum-only UIA over five victim batches and
three seeds with `T in {1,2,5,10,20,50}`. Log rank, CV,
`Spearman(final loss, ||b_i||)`, predicted `B_eff`, and the full
`lambda_i/mp_edge` spectrum. Still no ICA, Stage 2, FIA, or `--refine-A`.

Resolved `B_eff` validation: after robustness, ICA + Hungarian matching was run
only for `n in {16,32,64}`, `bits in {2,3,4,8}`, `T in {1,10}`, three seeds,
plus `scale_granularity in {per-tensor, per-bucket}` at 4-bit for
`n in {16,32}`. It reports predicted `B_eff` against measured recovered count
(`cos > 0.9`) and the full source-cosine values. This is the main MP prediction
contribution evidence.

The `m5_uia_beff_validation` prediction count is now fixed by direct recount
from `quantization_aware_cpa/results/m5_uia_beff_validation/summary.csv`:
`81/96` exact, MAE `0.219`, max error `2`. No alternate-count source was found
under `quantization_aware_cpa/results`. Per-row is
strong (`62/72` exact, MAE `0.181`), per-bucket is strongest among coarse
granularities (`11/12` exact, MAE `0.083`), and per-tensor is the most
pessimistic/noisy (`8/12` exact, MAE `0.583`).

Core per-row T split (`n in {16,32,64}` x `bits in {2,3,4,8}` x three seeds):

| subset | conditions | exact | MAE |
|---|---:|---:|---:|
| T=1 (= FedSGD) | 36 | 30 | 0.194 |
| T=10 (= FedAvg) | 36 | 32 | 0.167 |
| overall | 72 | 62 | 0.181 |

Paper sentence: 예측식은 96 조건 중 81 에서 정확했다 (MAE 0.219). FedSGD 에
해당하는 T=1 전체 부분격자에서는 39/48 이 정확했다. 이 중 per-row core
부분격자에 한정하면 30/36 이 정확했다.

T=1 에서 UIA 와 GIA 는 rel_l2 = 0.0 으로 동일하므로 UIA 격자는 GIA 를 특수
케이스로 포함한다. 예측식 검증은 T in {1, 10} 양쪽에서 수행되었다. The
scatter is square-aspect, separates `T=1 (= FedSGD)` and `T=10 (= FedAvg)` by
marker, and is archived with `run_tag="m5_beff_validation_T1_split"`.

## 0.5 v13 premise validation and round dependence

Upstream 4-bit CPA/FIA was checked by inserting quantization only at upstream
gradient collection. The upstream ICA, FIA, metrics, and matching code were
otherwise left unchanged.

| row | batch size | quantization | LPIPS |
|---|---:|---|---:|
| upstream CPA/FIA | 16 | 4-bit stochastic per-row | 0.5340 |
| local naive | 8 | 4-bit stochastic per-row | 0.5439 |
| upstream CPA/FIA | 32 | 4-bit stochastic per-row | 0.5615 |

Footnote: the batch sizes differ (`B=8/16/32`), so this is trend agreement, not
same-condition equality. The local naive row sits between the two upstream
rows, consistent with the upstream degradation as batch size increases.

All GIA/FedSGD reconstruction numbers before `m6_round_dependence` use the
round-0 ImageNet-pretrained VGG-16 checkpoint. Tables must now log the attacked
round or checkpoint explicitly. Threat-model note: single transmission observed
-> reported leakage is a lower bound.

`m6_round_dependence` trains only the FC layers, with VGG-16 convolutional
features frozen and cached. Training data and measurement data are disjoint:
training uses ImageNet `train` split batches `0-4`, while measurement uses
held-out ImageNet `val` split batches `0-4`. The optimizer is SGD with
`lr=1e-3`, train batch size `32`, five train batches cycled, classifier
dropout disabled/eval-mode, `n=32`, 4-bit stochastic per-row quantization, and
no ICA, Stage 2, FIA, GMA, or `--refine-A`.

| round | train loss | measurement loss | CV(||a_i||) | min/max ||a_i|| | pred. B_eff | sigma2 | mp_edge | lambda_max | Spearman(loss, ||a_i||) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7276 | 1.1856 | 1.102 | 5.86e-5 | 21.2 | 1.330e-7 | 2.622e-7 | 3.061e-4 | 0.958 |
| 5 | 0.6881 | 1.1833 | 1.104 | 5.79e-5 | 21.2 | 1.333e-7 | 2.628e-7 | 3.070e-4 | 0.959 |
| 20 | 0.5818 | 1.1768 | 1.109 | 5.57e-5 | 21.2 | 1.336e-7 | 2.634e-7 | 3.089e-4 | 0.960 |
| 50 | 0.4171 | 1.1659 | 1.118 | 5.08e-5 | 20.6 | 1.348e-7 | 2.658e-7 | 3.139e-4 | 0.962 |
| 100 | 0.2448 | 1.1529 | 1.134 | 4.47e-5 | 20.2 | 1.371e-7 | 2.703e-7 | 3.218e-4 | 0.963 |

Interpretation status: hold the learning-stage-insensitivity claim. Train loss
falls `0.7276 -> 0.2448`, while predicted `B_eff` changes only `21.2/32 ->
20.2/32` and Spearman changes `0.958 -> 0.963`; however, measurement loss moves
only `1.1856 -> 1.1529` (`-2.8%`). The spectrum decomposition shows no
shrinking-scale cancellation: `sigma2` rises `1.330e-7 -> 1.371e-7`,
`mp_edge` rises `2.622e-7 -> 2.703e-7`, `lambda_max` rises `3.061e-4 ->
3.218e-4`, and `lambda_max/mp_edge` stays nearly preserved (`1167.2 ->
1190.4`). This is best read as a near-unchanged measurement state, not as an
established learning-stage invariance result. The full `lambda_i/mp_edge`
vectors are in
`quantization_aware_cpa/results/m6_round_dependence/round_lambda_vectors.csv`;
the summary figure is archived under
`quantization_aware_cpa/results/m6_round_spectrum_decomposition/`.

Round-0 train/val gap audit: `training_loss_mean` is a checkpoint probe, not a
moving average, and round 0 is measured before any optimizer step. Train and
measurement batches are disjoint images from ImageNet `train`/`val`, but cached
labels have the same batchwise class sequence and full label overlap, so class
distribution does not explain the `0.7276` vs `1.1856` gap. The remaining
log-visible explanation is split/image difficulty; per-train-batch checkpoint
losses were not logged without recomputation.

At every checkpoint, UIA/FedAvg `T=10` updates were saved for later round
dependence checks without rerunning FC training:
`quantization_aware_cpa/results/m6_round_dependence/uia_T10_dumps/`
contains 25 dumps and `uia_T10_dump_index.csv` records their protocol fields.

## 0. Current state — what exists, what does not

Do not rebuild what is already verified. Do not assume anything else works.

| Component | State |
|---|---|
| Per-row quantizer, both rounding modes | **verified** (Gate 0 passed, bits 2/3/4/8) |
| Hook-based `A_true` / `S_true` extraction | **verified** |
| `Y == A_true @ S_true` vs autograd | **verified** |
| LP feasibility / `gamma` diagnostics | **works** |
| Stage 1 (debias + ICA) | **BUGGY — see §2** |
| Milestone -1 (CPA ICA port, fp32 repro) | **NOT DONE — see §1** |
| **Stage 2 solver** | **implemented for hinge-only and L1-prior embedding solve** |
| `A_oracle` | **implemented; hinge-only and L1-prior runs exist** |
| `bplus` | **implemented with the same L1 prior budget and no coupling** |
| `naive`, `debias`, `A`, `gma` | **do not exist** |
| Stage 3 | does not exist (intentional) |

The table above is legacy v3 scaffolding where it conflicts with sections
0-0.2. Current code has Stage 1 implemented as CPA ICA + MP truncation +
LS-scaled `A_hat`; Milestone -1 is passed; `naive`, `debias`, `A`,
`A_oracle`, and `bplus` rows exist. `gma` remains an upstream image-gradient
baseline outside the embedding-only harness.

### The three existing rows in `results/runs.jsonl` are void

They were produced with the §2 bug. Every derived number (`rho`, `gamma`,
`source_cos`, `cond_A_hat`, infeasible-LP counts) is invalid. Tag them and never
mix them into paper numbers:
```
run_tag: "void_prewhitening_bug"
```
Add a required `run_tag` field to the schema. Paper runs use `run_tag: "v3"`.

### What the first run did establish (worth keeping in mind)

- Failure is **bimodal**, not graceful: 6/8 sources at cosine 0.995-0.999, 2/8
  at 0.001-0.017, in all three batches. The mean (0.75) hides this entirely.
- The two dead sources correspond to the two smallest covariance eigenvalues,
  and their recovered scales collapsed to ~0.002-0.03.
- Even the weakest eigenvalue sits ~25x above the quantization noise floor, so
  **the information is present**; the loss is algorithmic.
- 64-79% of sampled columns had an **empty** feasible set, i.e. `A_hat` is wrong
  enough to make the constraint system inconsistent.
- `cond(A_hat)` was 11-16 while `cond(A_true)` was 92-3131: the estimate cannot
  represent an ill-conditioned true mixing.

---

## 1. BLOCKING DECISION — settle Milestone -1 first

The first run used `sklearn.FastICA` (`fun=logcosh, max_iter=1000, tol=1e-4`).
That is **not** CPA's method. CPA optimizes a custom unmixing objective
(non-Gaussianity + independence penalty + TV prior), not off-the-shelf FastICA.
No upstream fp32 reproduction numbers exist yet either.

This must be settled **before** Milestone 2, because swapping the ICA later
invalidates `A_hat`, `S_hat`, every diagnostic, every Stage-2 result, and every
tuned hyperparameter.

**Required path:**

1. Port the upstream repo (`facebookresearch/cocktail_party_attack`) to
   py3.10+/torch>=2.7+cu128. Its `python=3.8` env cannot run on sm_120
   (RTX 5060 Ti, Blackwell). Verify
   `torch.cuda.get_device_capability() == (12, 0)`.
2. Read their ICA implementation. **Write down the exact objective and optimizer
   as implemented**, not as described in the paper.
3. Reproduce upstream CPA at **fp32, B=32**, unmodified. Compare `source_cos`
   and image metrics against the paper's reported values.
4. Replace `sklearn.FastICA` with CPA's ICA in our Stage 1, keeping our debiased
   whitening in front of it.

Do step 4 **after** the §2 fix, so "CPA ICA is better" and "whitening was broken"
do not get confounded.

If you believe keeping `sklearn.FastICA` is preferable, **stop and report the
argument** — do not decide unilaterally. A baseline weaker than published CPA
makes the entire comparison unfalsifiable.

---

## 2. Bug fixes to apply

### 2.1 Whitening (primary suspect for the two dead sources)

`stage1.warnings` contained:
```
"Ignoring n_components with whiten=False."
```
`whiten=False` means the ICA assumes its input is already white. If Stage 1 only
**projects** onto the top-`B` eigenvectors without rescaling, the input is not
white — the eigenvalue spread was 128:1 — and the lowest-variance directions are
lost. That is exactly the observed failure.

Correct Stage 1:
```
C = (Y_hat @ Y_hat.T) / d                      # (4096, 4096)
C = C - torch.diag(delta**2 / 12)              # DIAGONAL, only if stochastic
lam_deb, P = top_B_eigh(C)                     # lam_deb: (B,), P: (B, 4096)
sigma2  = (delta**2 / 12).mean()
mp_edge = sigma2 * (1 + sqrt(N / d))**2
keep    = lam_deb > c_mp * mp_edge             # c_mp default 3.0, config logged
lam_k   = lam_deb[keep]                        # (B_eff,)
P_k     = P[keep]                              # (B_eff, 4096)
Y_red   = (lam_k ** -0.5)[:, None] * (P_k @ Y_hat)   # (B_eff, 25088)
A_hat_red, S_hat = ICA(Y_red)
A_hat = P_k.T @ (lam_k[:, None] ** 0.5 * A_hat_red)  # lift back to (4096, B_eff)
```
The `lam ** -0.5` scaling is required, but absolute eigenvalue clamps are forbidden.
Directions below the Marchenko-Pastur noise-relative threshold are dropped
before whitening so whitening cannot amplify them into all retained coordinates.
Assert `cov(Y_red) ≈ I` in the `B_eff` subspace (off-diagonal < 1e-3) before
calling ICA.

### 2.2 Debias must be a diagonal, not a scalar

The logged `debias_variance` was a single scalar. Scales are **per-row**, so the
quantization noise covariance is `diag(delta_j^2 / 12)`, not
`mean(delta^2)/12 * I`. Row-to-row gradient magnitude varies a lot in this FC
layer, and the error falls hardest on the weak components — the ones that died.

### 2.3 The `bits16` gate was vacuous

It ran with `deterministic` rounding, where debias is disabled, so `debias` and
`naive` are identical by construction and `relative_l2 = 0.0` is automatic. It
tested nothing. **Re-run with `stochastic` only**, and assert
`relative_l2 < 1e-3` rather than `== 0`.

### 2.4 Verify the factorization gate is not self-comparing

`factorization_matches_autograd` reported `max_abs_err` of exactly `0.0`.
Comparing the hook product against autograd's `G` should leave float error
around `1e-7`, not exactly zero. Confirm the two sides are computed
independently; if not, fix the test.

### 2.5 Cache invalidation

Changing any Stage-1 parameter must invalidate cached `A_hat` / `S_hat`. Hash the
Stage-1 config into the cache key.

---

## 3. Exact shapes (unchanged — already verified)

Model: `torchvision.models.vgg16(weights=IMAGENET1K_V1)`, eval, frozen.
Attacked layer: `model.classifier[0]` = `Linear(25088, 4096)`.

```
Y = A @ S
Y : (N, d) = (4096, 25088)
A : (N, B) = (4096, B)      # backprop deltas, columns indexed by sample
S : (B, d) = (B, 25088)     # per-sample embeddings = ICA sources, S >= 0 exactly
N = 4096 mixtures,  d = 25088 ICA samples per source
```

Ground truth via hooks on `classifier[0]`: forward hook `input[0]` -> `S_true`;
full backward hook `grad_output[0]` -> `A_true.T`.

Large `B`: accumulate over microbatches — the aggregate gradient is a sum, so
this is exact. Rescale if the loss is a batch mean. Hooks must accumulate in the
same order.

**Batches must have all-distinct labels** for every gate run. Add
`data.require_distinct_labels: true`, assert it, and log the label list.
`cond(A_true) = 3131` in one of the void runs suggests this was not enforced;
duplicate labels correlate the columns of `A` and break ICA independently of
quantization.

---

## 4. Quantizer (verified — do not change)

Per-row bucketed uniform affine; one scale per row `j` of `Y`. A global scale is
degenerate at 4 bits and does not match how QSGD-style defenses bucket.

Invariant, holds for every entry:
```
(Y - Y_hat).abs() <= delta[:, None] / 2 + 1e-6
```

Both rounding modes must run and are **not** equivalent:
- `stochastic` — error unbiased, independent, uniform; the §2.2 debias is exact.
- `deterministic` — error correlates with the signal; debias becomes an
  approximation. **The Stage-2 hinge stays exactly valid in both cases.**

That asymmetry (our constraint is assumption-free; debiasing is not) is a
**result to report**, not a bug to hide.

---

## 5. Method

### Stage 1 — estimate A

As corrected in §2.1/§2.2, first debias the covariance diagonally, compute the
top requested `B` eigenvalues, determine `B_eff` with the MP threshold
`lam_deb > c_mp * mp_edge`, and whiten only the retained directions. Use CPA's
ICA (§1) in place of FastICA after this folded-in Milestone-1 fix.

Log `cond(A_hat)`, `cond(A_true)`, requested `B`, retained `B_eff`, `sigma2`,
`mp_edge`, `c_mp`, all pre-truncation `lam_deb` values, discarded component
indices, kept `lam/mp_edge` ratios, and the debias diagonal summary on every run.

### Stage 2 — consistency solve

Column-separable. Use dimensionless residuals and normalize each feature column by the pinv initialization scale.

```
s_scale_k = max(||s_init_k||_inf, eps_scale)
u_k       = s_k / s_scale_k
r_j(s)   = (a_j . s - Y_hat[j,k]) / delta_j
H(s)     = sum_j max(0, |r_j(s)| - 0.5)^2
```

Two-phase solve for each column `k`:

```
Phase 1: s0 = argmin_{s >= 0} H(s)
         h0 = H(s0)

Phase 2: minimize_{s >= 0}
            lam_R * ||s / s_scale_k||_1
          + mu * max(0, H(s) - h0 - eps)^2
```

Default debug values: `mu=1e3`, `eps=1e-3`. The L1 prior is tuned on exactly
one held-out batch and then fixed for all reported batches; after
dimensionless normalization the current debug tuned value is `lam_R=300`.

Batched implementation, chunked over `d`:
```python
A_scaled = A / delta[:, None]
Y_scaled = Y_hat / delta[:, None]
L        = 2.0 * torch.linalg.matrix_norm(A_scaled, ord=2) ** 2
L        = L * max_k(s_scale_k) ** 2
eta      = 1.0 / L

# Phase 1 uses projected gradient with optional FISTA momentum.
# Phase 2 uses the same eta, L1 prox in u-space, and a guard that accepts
# only columns with H(s) <= h0 + eps, so L1 cannot buy sparsity by selling
# quantization consistency.
```
- Chunk `C` so the `(4096, C)` residual fits comfortably; `C = 4096` is ~67MB.
  Whole stage should be minutes, not hours.
- Record per-column final hinge to compute the §7 metrics.
- Record per-column `H(s0)`, `H(s_final)`, raw `||s||_1`, normalized
  `||s/s_scale||_1`, and Phase 1 / Phase 2 consistency rates.
- Hinge-only Stage 2 was confirmed effectively no-op under the old dimensional
  hinge. The dimensionless Phase 1 moves more, but weak sub-threshold sources
  can still remain below the recovery threshold.

Feasibility note: the feasible set was **empty for 64-79% of columns** in the
void runs. The hinge is a min-violation formulation, so it still runs — but the
solution is then pulled toward a wrong `A_hat`. Hence `--refine-A` (below)
matters earlier than originally planned.

Optional flag `--refine-A`, **off by default**: an empty feasible set means
`A_hat` is wrong, so residual infeasibility is a usable objective. Alternate a
few `S`-steps with one small gradient step on `A_hat` against the same hinge.
Never jointly with Stage 3.

### Stage 3 — end-to-end (LOCKED — do not build)

Gated behind Milestone 3. `A_hat` frozen, warm-started from `S*`:
```
minimize_x  - sum_i CS(f(x_i), S_star[i])
            + lam_tv * sum_i TV(x_i)
            + lam_q  * sum_{j,k} hinge(|(A_hat @ f(x))[j,k] - Y_hat[j,k]|, delta_j/2)^2
```

---

## 6. Baselines — build in this order

Implement `A_oracle` **first**. It uses `A_true`, so it is independent of Stage-1
quality and needs only the Stage-2 solver. It is the method's upper bound: if it
fails, the other baselines are wasted work.

| order | id | description |
|---|---|---|
| 1 | `A_oracle` | Stage 2 given `A_true` — upper bound |
| 2 | `bplus` | decoupled relaxation (see below) |
| 3 | `A` | **ours**: corrected Stage 1 + Stage 2 |
| 4 | `debias` | Stage 1 with debias, no Stage 2 |
| 5 | `naive` | Stage 1 without debias, no Stage 2 |
| 6 | `gma` | upstream GMA, unmodified |

### `bplus` — and its expected degeneracy

Per-source Mahalanobis ball, no coupling:
```
sigma_i^2 = sum_j (delta_j^2 / 12) * w_ij^2      # w_i = row i of the unmixing matrix
bplus: minimize lam_R * ||s/s_scale_k||_1
       s.t.  |s_i - S_hat[i,k]| <= c * sigma_i,  s >= 0
```

Without a prior this nearly collapses to `debias`. With the same fixed
dimensionless L1 prior and the same iteration budget as Stage 2, it becomes the
valid decoupled baseline. It still may not receive coupling through `A`.

The comparison now measures the value of coupled quantization constraints over
the decoupled interval relaxation under the same prior.

**Do not give `bplus` coupling to make it "fairer."** Keep the identical L1
prior, tuning budget, and iteration count when comparing against Stage 2.

---

## 7. Metrics

### Mandatory reporting changes from v2

1. **`source_cos` must be logged per-source, not only as a mean.** The mean hid a
   6/8-perfect + 2/8-dead structure in the first run. After MP truncation, log
   the full length-`B_eff` matched vector plus `n_sources_recovered` (count with
   `|cos| > 0.9`), `recovered_over_beff`, and the honest whole-batch
   `recovered_over_B`. Also log `sub_threshold_recovered`, the number of truth
   sources recovered despite `lam/mp_edge < c_mp`; this is structurally zero for
   `bplus`.
2. **`feasible_column_fraction` and `infeasible_column_fraction` are first-class
   metrics**, not byproducts of the `gamma` diagnostic. They are the most direct
   measure of `A_hat` quality.
3. **`gamma` may never be reported alone.** It is measured only over feasible
   columns, so it is optimistically biased. Always emit it next to
   `feasible_column_fraction`.
4. **Log per-source `||a_i||` and per-sample loss and label.** Test whether the
   dropped/unmatched sources are the small-gradient, low-loss, well-classified
   samples. If so, that is a reportable finding about *which* samples leak.
5. **Log MP truncation state.** Include `B_eff`, `sigma2`, `mp_edge`, `c_mp`,
   all requested `lam_deb/mp_edge` ratios, and discarded component indices.
6. **Log Stage-2 phases.** Include per-column `H(s0)`, `H(s_final)`, raw and
   normalized L1 before/after, Phase 1 / Phase 2 consistency rates, and the
   Phase 2 guard rejection fraction.

### Constraint satisfaction
- `consistency_rate` — fraction of `(j,k)` inside their cell
- `residual_violation` — final mean hinge
- `feasible_column_fraction`, `infeasible_column_fraction`

### Recovery quality (Hungarian matching on `|cosine|`)
- `source_cos_per_source` (length `B_eff`), `source_cos` (mean over `B_eff`),
  `n_sources_recovered`, `recovered_over_beff`, `recovered_over_B`,
  `sub_threshold_recovered`
- `scale_err` — mean `| ||s_hat_i|| / ||s_true_i|| - 1 |` (only `A` can report this)
- image metrics — Stage 3 only

Judge success in **embedding space** (`source_cos_per_source`), not image space.
ImageNet 224x224 feature inversion is the weakest link even upstream; image
metrics cannot separate "Stage 2 failed" from "FIA failed."

### Diagnostics
`rho`, `gamma` (+ feasible fraction), `mean_axis_extent`, `cond_A_hat`,
`cond_A_true`, `cond_A_true_matched`, all requested `B` eigenvalues, `B_eff`,
`mp_edge`, debias diagonal summary, `||a_i||` per source, per-source loss and
label. For Stage 2, log Phase 1 / Phase 2 `H`, consistency rates, raw and
normalized L1, and the Phase 2 consistency-guard rejection fraction.
Define `mean_axis_extent` as the mean over feasible sampled columns and axes of
`(max_i(k) - min_i(k)) / max(||s_hat[:, k]||_2, 1e-12)`. Define the gamma
denominator as `normalized_ica_rmse =
sqrt(mean_{i,k}(((s_hat[i,k] - s_true[i,k]) / max(||s_hat[:,k]||_2, 1e-12))^2))`,
and log `gamma = mean_axis_extent / normalized_ica_rmse`.
Compute LP diagnostics on 512 sampled columns; for `B > 64` use 32 random
directions instead of `2B` axis directions.

### Diagnosis table — use this instead of blind tuning

| consistency | quality | diagnosis | what to change |
|---|---|---|---|
| high | high | success | nothing |
| high | low | wrong point inside the cell | add `r(S)` / TV |
| low | low | `A_hat` wrong, or empty feasible set | Stage 1, `--refine-A` |
| low | high | **implementation bug** | re-run Gate 0 |

---

## 8. Milestones — STOP and report at each

Debug config: `B=8, bits=4, stochastic, distinct labels, seed=0`.
First **reportable** config: `B=32`.

### Milestone -1 — settle the ICA and reproduce fp32 (BLOCKING)
Per §1: port env, verify sm_120, document CPA's real ICA objective, reproduce
upstream fp32 at `B=32`, then swap CPA's ICA into our Stage 1.

**STOP. Report reproduction numbers, the ICA objective as implemented, and
`get_device_capability()`.**

### Milestone 0 — correctness (re-run)
1. Quantizer invariant, both modes, bits {2,3,4,8}. *(already passing)*
2. Oracle test: hinge at `(A_true, S_true)` is exactly 0. *(already passing)*
3. `Y == A_true @ S_true` vs autograd — re-verify per §2.4.
4. `bits16` debias-vs-naive — re-run with **stochastic** per §2.3.
5. **New:** `cov(Y_red) ≈ I` after whitening (off-diagonal < 1e-3).
6. **New:** batch labels are all distinct.

**STOP. Report all six.**

### Milestone 1 — re-run diagnostics after the §2 fixes
3 batches with mandatory MP truncation folded in. Report requested `B`,
retained `B_eff`, all 8 `lam_deb/mp_edge` ratios, `source_cos_per_source`
(length `B_eff`), `n_sources_recovered`, `recovered_over_beff`,
`recovered_over_B`, `rho`, `gamma`, `feasible_column_fraction`, `cond_A_hat`,
`cond_A_true`, `||a_i||`, loss, and label per source.

Decision:
- 8/8 recovered **and** feasible fraction high -> go straight to Milestone 2.
- 8/8 recovered but feasible fraction still low -> run `A_oracle` feasibility
  to isolate residual `A_hat` error.
- Still weak after MP truncation -> Stage 1 remains the blocker. Do not advance
  to Stage 2.

**STOP. Report.**

### Milestone 1b — `B_eff` truncation (MANDATORY, folded into Milestone 1)
Drop components below the MP threshold before whitening, then re-measure the
feasible fraction and report both `recovered_over_beff` and `recovered_over_B`.
If the dropped/unmatched samples are low-loss or small-gradient samples, report
that as the mechanism and keep `c_mp` as a Milestone-4 ablation axis.

**STOP. Report.**

### Milestone 2 — does the method work at all?
Build the Stage-2 solver, then run `A_oracle` vs `bplus` on 5-8 batches. The
effect size here is "recovered vs not," so 8 batches suffice; needing more means
it is not a result.

**If `A_oracle` does not clearly beat `bplus` on `source_cos_per_source` /
`n_sources_recovered`, stop. Do not build Stage 3.**

Then run `A` and report the gap `A_oracle - A` (the `A_hat` estimation loss).

**STOP. Report the 6-row table** (`gma`, `naive`, `debias`, `bplus`, `A`,
`A_oracle`) with per-source values, not just means.

### Milestone 3 — is Stage 3 needed?
Feed `S*` from Stage 2 into upstream **unmodified** feature inversion (no
consistency term). If that captures the whole gain, Stage 3 is unnecessary and
the paper is simpler. Build Stage 3 only if measurable headroom remains.

**STOP. Report.**

### Milestone 4 — grid (only after -1..3 pass)
Milestone 4 is now split into a reduced two-phase grid. Before running any
ICA/Stage-2/FIA grid cell, run the MP prediction pass (`run_tag:
v10_mp_prediction_grid`) using only collected gradients, quantization, debiased
covariance, and eigenvalues. This gives `lambda_i/mp_edge` spectra and `B_eff`
for selecting boundary cells.

Phase A is the broad one-seed scan: `bits={2,3,4,6,8,fp32}`,
`B={32,128,512}`, per-row stochastic quantization, plus
`scale_granularity={per-tensor,per-bucket}` for `bits={3,4}` and
`B={32,128}`. B=512 must be emitted for both distinct and non-distinct label
selection to isolate duplicate-label conditioning effects. Deterministic
rounding is excluded from Phase A except for later spot checks.

Phase B fills only cells near an MP boundary or image-metric transition with
five seeds. Full `{32,64,128,256,512,1024}` and deterministic sweeps are
reserved for follow-up once Phase A/B identifies the boundary region.

- `bits`: {2, 3, 4, 6, 8, fp32}
- `B`: Phase A {32, 128, 512}; full CPA range is deferred
- `rounding`: stochastic in Phase A; deterministic only as spot checks
- seeds: 1 in Phase A, 5 in Phase B boundary cells
- scale granularity: {per-tensor, per-row, per-bucket}
- ablations: debias on/off; scalar-vs-diagonal debias; `N` row-subsample
  {100%, 50%, 25%}; duplicate-label batches (sweeps `cond(A)`); `c_mp` MP
  truncation threshold. Do not include hinge-only `--refine-A` as a grid axis;
  it is a recorded negative diagnostic, not a candidate method.

Primary grid metrics are image LPIPS and `Spearman(per_sample_loss,
per_image_lpips)`. Embedding cosine thresholds are auxiliary only; the retained
subspace showed that `cos=0.9962` vs `0.9990` can move LPIPS (`0.4596` vs
`0.4511`) while `cos > 0.9` counts hide the difference. Prepare figures for
the `(B, bit)` heatmap with MP boundary prediction, `lambda/mp_edge` vs cosine,
loss vs LPIPS scatter with CI, and the original/fp32/quantized image grids.

---

## 9. Hard prohibitions

Violating any of these invalidates the experiment.

1. **Never use a straight-through estimator on `Q`.** STE collapses the
   consistency term into `||g(x) - Y_hat||^2`, which *is* naive dequantization.
   The hinge is the method. There is nothing to differentiate through `Q`.
2. **Do not treat the hinge dead zone as a bug.** Zero gradient inside the cell
   is correct — the observation carries no information there. The hinge is a
   barrier, not an attractor, so never remove the CS/prior term in Stage 3.
3. **Do not give `bplus` coupling.** Its degeneracy is the argument (§6).
4. **Do not modify CPA's ICA objective, FIA, metrics, or matching code.**
5. **Do not add priors, TV, or extra regularizers before Milestone 2.**
6. **Do not report `source_cos` as a mean only.** Per-source vector is mandatory.
7. **Do not report `gamma` without `feasible_column_fraction`.**
8. **Do not tune only `A`.** Equal budget and equal iteration count for all
   baselines.
9. **Do not use a global quantization scale**, and do not use a scalar debias.
10. **Do not report `sign_accuracy` as a contribution** — `S >= 0` already fixes
    sign on VGG16. Report `scale_err`.
11. **Do not pass the full (4096, 25088) matrix to sklearn.** Reduce to `B` dims
    first; full-SVD whitening in float64 will exceed 16GB system RAM.
12. **Do not skip Milestone -1 or Milestone 0.** A wrong `delta`, an unwhitened
    input, or a non-CPA ICA all produce plausible bad numbers, not a crash.

---

## 10. Engineering constraints (16GB VRAM / 16GB RAM)

| item | size | note |
|---|---|---|
| `Y` fp32 (4096x25088) | 411 MB | one copy, on GPU |
| `C` (4096x4096) | 67 MB | fine |
| VGG16 full gradient | ~550 MB | fine |
| Stage-2 chunk residual | ~67 MB | chunk over `d` |

- Config-driven single entry point:
  `python -m qacpa.run --config configs/spike.yaml`
- Fully seeded and deterministic given a seed; log the seed.
- **Required schema fields:** `run_tag`, `git_sha` (currently `"unavailable"` —
  fix this), `schema_version`.
- **Cache as `int8` q + `delta`**, never fp32 `Y_hat` (~822MB/config, ~25GB over
  the grid; int8 + scales is ~103MB). Recompute `Y_true` from the seed.
  Cache keys must hash the Stage-1 config (§2.5).
- One row per run appended to `results/runs.jsonl` with full config, every metric
  in §7, every diagnostic, wall time. Figures are generated later from this
  file — **do not build plotting now.**
- Log every metric **per victim batch**, never only the mean.

---

## 11. First deliverable

Milestone -1 only:
1. Env ported; `torch.cuda.get_device_capability()` output.
2. CPA's ICA objective and optimizer, as actually implemented upstream.
3. Upstream fp32 CPA reproduced at `B=32`, compared against the paper.

Nothing else. Report and wait.
