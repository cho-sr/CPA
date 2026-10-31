# STATUS

Latest task: **Top-k10 uniform random-fill experiment COMPLETE. All80 images evaluated after20 candidates per original batch, FIA5k screening/top3 to total25k. Results and16:9 reconstruction PNG verified.**

## Top-k10 random fill, same80, R20 screen5k/top3 final25k

- Authorized2026-09-12; user explicitly selected **10% only** and **U(-tau,tau)**.
- Same five actual packets, same80 original images, n16/T10 and frozen upstream CPA/FIA.
- New random fills only at dropped coordinates; tau from retained packet values. No new victim training or collection and no additional communication.
- Whole-candidate selection by recompressed-update relative L2, public-model predicted labels; no ground-truth ranking. All20 candidates screen5k, top3 continue to total25k, then one final whole16-image candidate per batch is selected by the same observable score.
- State: **COMPLETE**,2026-09-12 21:37:25 KST. Launched02:49:25 KST through hidden CMD PID15420, Python PID5740; elapsed about18h48m. All100 candidates completed FIA5k and all15 finalists completed total25k. Final whole-batch candidate IDs for batches0..4:10,9,9,5,10.
- Verified all80 means: LPIPS0.6299886767, SSIM0.2610037789, PSNR21.5103999853. Same Top-k10 zero-fill reference:0.6323260389 /0.2631218264 /21.5305386782. LPIPS slightly lower; SSIM/PSNR slightly lower; no consistent improvement across image metrics. Source cosine>0.9 remains52/80, an embedding diagnostic.
- All80 original identities/order, per-image means, completed output hashes, all100 screen counts,15 finalist iteration counts and observable score selections rechecked after completion.
- [All80 original/reconstruction16:9 PNG](../quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/all80_original_vs_reconstruction_16x9.png); [report](../quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/report.md).
- User-requested zero/random comparison exported: metric graph, all80 paired reconstruction PNG, five Original/Zero/Random batch PNGs, paired per-image/per-batch CSVs, manifest and ZIP. All figures3840x2160; image identities, packet hashes and400 rendered tiles checked. Random-minus-zero means: LPIPS-0.00233736, SSIM-0.00211805, PSNR-0.02013869dB; random improves47/80,39/80,44/80 respectively (descriptive paired counts). [Comparison report](../quantization_aware_cpa/exports/zero_vs_random_topk10_same80/report.md); [all files ZIP](../quantization_aware_cpa/exports/zero_vs_random_topk10_same80/zero_vs_random_topk10_comparison.zip).
- Validation:52 focused tests passed; staged Direct FIA matched original continuous execution, including Adam/pixels/cosine states. All five real filled observations recompressed to exactly the original packet bytes; GPU classifier-replay ranking path verified. Input/config/code/weights fingerprints frozen.
- [Setup verification](../quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/setup_checks/verification.json); [launch record](../quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/launch.json).
- [Protocol and CMD usage](../quantization_aware_cpa/docs/RANDOM_FILL_SAME80.md).
- [Execution status](../quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/execution_status.json).
- Compute differs from existing one-attempt zero fill by user request; do not attribute differences solely to filling. No other keep fractions/control arms authorized.

## Top-k30/20/10% sparse CPA + FIA, same historical80

- Authorized2026-09-11: retain30%,20%,10% of FC1 coordinates, in that order.
- Same cached80 images and n16x5/T10 round0 updates; unchanged CPA/FIA25000, seeds and post-FIA matching.
- State: **COMPLETE**,2026-09-11 22:56:46 KST. All15 batches and all80 images per condition completed; queue elapsed8h17m28s. Original launch was14:39 KST via CMD PID11168. Automatic per-condition reports and final comparison completed successfully.
- All80 means (LPIPS / SSIM / PSNR):30%0.5098320577 /0.3140844309 /22.1371316195;20%0.5536050621 /0.2906016125 /21.8719979763;10%0.6323260389 /0.2631218264 /21.5305386782. Source cosine>0.9 counts66/60/52 of80 are embedding diagnostics, not image-success counts.
- [Verified final comparison](../quantization_aware_cpa/results/cpa_fia_topk30_20_10_same80_B16/report.md).
- User-requested16:9 exports:50/30/20/10%, each includes all80 in original order and160 Original/Reconstruction panels,3840x2160. Source PNG/CSV/manifest hashes and every resized tile were verified; no synthesized pixels or color adjustment. [Four PNGs in one ZIP](../quantization_aware_cpa/exports/topk_same80_reconstruction_16x9/topk50_30_20_10_reconstruction_16x9.zip); [export manifest](../quantization_aware_cpa/exports/topk_same80_reconstruction_16x9/export_manifest.json).
- Validation:184 focused CPU tests passed; attack/evaluation function ASTs identical to the frozen50% runner. All15 real packets and their original80 identities/source-update hashes verified before CUDA launch. CUDA capability(12,0) confirmed.
- Packet bytes per batch for30/20/10%:136,157,624 /95,053,444 /53,949,264. FC1 packet savings66.875% /76.875% /86.875%, respectively. Actual nonzero values are pruned: relative L2 error ranges0.1366%-0.8067% /1.8792%-4.5131% /11.4079%-16.1296%.
- [Setup verification](../quantization_aware_cpa/results/cpa_fia_topk30_20_10_same80_B16/setup_verification.json); [15-packet audit](../quantization_aware_cpa/results/cpa_fia_topk30_20_10_same80_B16/sparsification_input_audit.json).
- Separate configs and output directories preserve all completedTop-k50 and packed4 artifacts.
- [Protocol and CMD usage](../quantization_aware_cpa/docs/TOPK30_20_10_SAME80.md).
- [Queue status](../quantization_aware_cpa/results/cpa_fia_topk30_20_10_same80_B16/queue_status.json).

## Top-k 50% sparse CPA + FIA, same historical80

- Run tag: `cpa_fia_topk50_same80_B16`; explicitly authorized 2026-09-11.
- Exact same80 cached n16/T10 updates; magnitude Top-k within FC1, keep50%, FP32 values plus bitmap, receiver zero fill.
- Unchanged upstream CPA/FIA, 25,000 iterations each, CPA seed=batch, FIA seed42, post-FIA evaluation; no retraining or additional arms.
- State: **COMPLETE**, 2026-09-11 13:36 KST. All80 LPIPS 0.4863830283, SSIM 0.3344356938, PSNR 22.2877590895; source |cos|>0.9 is71/80 (embedding diagnostic, not an image-success count). Launch-to-report elapsed 2h47m39s. 60 focused CPU checks passed; all five packets verified, 218,365,984 bytes each.
- Per-image means and all80 identities were rechecked after completion; per-image/per-batch CSVs and six reconstruction PNG hashes match the summary. [Report and images](../quantization_aware_cpa/results/cpa_fia_topk50_same80_B16/report.md).
- Input audit: original nonzero fractions 38.2%-42.1%; Top-k50 drops only zero coordinates, with decoded relative L2 error0 in all five batches. The requested condition is retained as a lossless sparse representation of these updates.
- [Protocol and CMD usage](../quantization_aware_cpa/docs/TOPK50_SAME80.md).
- [Execution status](../quantization_aware_cpa/results/cpa_fia_topk50_same80_B16/execution_status.json).

## M10 same historical80 packed4 setup

- Run tag: `m10_cpa_fia_packed4_same80_B16`.
- Source: the exact M6/M8 images, IDs0-79 in original order, five cached n16/T10 actual updates.
- State: **COMPLETE**. All80 reconstructed; LPIPS 0.5732020233, SSIM 0.2889189334, PSNR 21.8309161901. Five real packed4 packets, 51,396,640 bytes each. Completion verified from the user-executed run.
- Validation: 90 focused CPU tests passed, including provenance, batch/packet identity, attack boundary, codec and full80 reporting.
- Separate entry point and output preserve the completed best10 condition.
- [Usage and execution commands](../quantization_aware_cpa/docs/M10_SAME80_PACKED4.md).
- [Execution status](../quantization_aware_cpa/results/m10_cpa_fia_packed4_same80_B16/execution_status.json).

<!-- M10_PACKED4_STATUS_BEGIN -->
## M10 selected best10 CPA + FIA with actual packed4

- State: **COMPLETE**; run tag `m10_cpa_fia_packed4_best10`.
- Historical M8 lowest-LPIPS10 of80; selected exploratory sample. One new n10/T10 client update from stored features.
- Only packed4 CPA+FIA; upstream algorithms and budgets unchanged. No FP32 or raw-row attack.
- [Execution status](../quantization_aware_cpa/results/m10_cpa_fia_packed4_best10/execution_status.json).
- All10: LPIPS **0.453650**, embedding cosine>0.9 **9/10**.
- [Report and figures](../quantization_aware_cpa/results/m10_cpa_fia_packed4_best10/report.md).
<!-- M10_PACKED4_STATUS_END -->

## Communication compression redesign (2026-09-10)

The user requested an audit of recent quantization and a redesign changing only
compression. [Audit and experiment design](docs/COMMUNICATION_COMPRESSION_REDESIGN_20260910.md)
and [machine-readable protocol](../quantization_aware_cpa/configs/m10_compression_only_B16.protocol.json)
are complete. The latter is a design artifact, not an executable runner config.

The recent stochastic path stores continuous FP32 additive-noise observations;
it does not implement an integer transmission codec. The new design uses
per-row 16-level stochastic integer quantization, packed two codes per byte,
with decoder-visible FP32 radii. Freeze each latest M8/M9 attack and compare
fresh FP32 versus actual packed4 observations of the same saved actual updates.
M6 FP32 cannot replace the new M8-style FP32 arm because whitening and pre/post-FIA
correspondence differ. Expected roughly8x savings apply to the attacked FC1
weight packet only. No new training, codec implementation or GPU run occurred
during this redesign. Historical results and locks are preserved.

Consolidated result review (2026-09-09): [EXPERIMENT_RESULTS_20260909.md](docs/EXPERIMENT_RESULTS_20260909.md).

Current naming: the user's new **FedAvg naive CPA** baseline includes CPA.
The control without source separation is explicitly **no-CPA**. Historical
`naive` keys denote the ported CPA + dither/dequant path, not no-CPA and not
a claim that every original upstream implementation detail was unchanged.

## No-CPA control validity correction (2026-09-09)

**Discard no-CPA control results: variable control failed, as confirmed by the user.**
All dependent CPA-contribution/IUT claims, normalization, LPIPS thresholds and
success counts are withdrawn. Raw artifacts and protocol locks are provenance
only. [Scope and unaffected sources](docs/NO_CPA_CONTROL_INVALIDATION_20260909.md).
The same80 FP32/4-bit client-update graph remains independent of this control.

<!-- M9_RAWROW_STATUS_BEGIN -->
## M9 FedAvg raw-row FIA

- State: **COMPLETE**; run tag `m9_fedavg_rawrow_dither4_B16`.
- Same80 n16/T10 cached M8 dither4 observations; fixed raw-row selection + abs + original FIA.
- No CPA, whitening, debias, MP, LS-A, embedding refinement, oracle fills, or new training.
- [Preregistered protocol](docs/FEDAVG_RAWROW_FIA_PROTOCOL.md).
- Original-image correspondence is evaluated only after FIA; this is a declared raw-row inversion baseline.
- The invalid historical M7 no-CPA results remain withdrawn.
- All80 results and landscape PNGs: [report](../quantization_aware_cpa/results/m9_fedavg_rawrow_dither4_B16/report.md).
- LPIPS **0.706844**, PSNR **21.353**, SSIM **0.2300**.
<!-- M9_RAWROW_STATUS_END -->

<!-- M8_NAIVE_STATUS_BEGIN -->
## M8 FedAvg naive CPA

- State: **COMPLETE**; run tag `m8_fedavg_naive_cpa_dither4_B16`.
- Direct original CPA and FIA classes; no LS-A, debias, MP or Stage2 calculation.
- Same80 cached n16/T10 dither4 observations; no new client training.
- Original-image/source correspondence is applied after FIA only.
- [Protocol and code entry points](docs/FEDAVG_NAIVE_CPA_PROTOCOL.md).
- All80 complete: LPIPS **0.556121**, PSNR **21.937**, SSIM **0.2980**, embedding cos>0.9 **53/80**.
- [Results and PNGs](../quantization_aware_cpa/results/m8_fedavg_naive_cpa_dither4_B16/report.md).
<!-- M8_NAIVE_STATUS_END -->

| Milestone | State | Note |
|---|---|---|
| -1 | PASS | Upstream CPA runs on Python 3.11 / torch 2.13.0+cu132; fp32 B=32 reproduction completed and matches/improves the paper LPIPS (`0.466` vs `0.483`). |
| 0 | PASSING | Static gates and batch gates run before Milestone 1 rows. |
| 1 | REPORT READY | `v3_mptrunc` completed on 3 debug batches. |
| 1b | FOLDED | MP `B_eff` selection is now part of Milestone 1 before whitening. |
| 2 | PASS | Dimensionless two-phase Stage-2 `A_oracle` beats `bplus` on sub-threshold recovery. Actual `A` now recovers all MP-retained sources after LS `A_hat` scaling. |
| 3 | REPORT READY | Full-n fallback report is complete and v8 retained-source precision diagnostics are recorded. |
| 4 | DEFERRED | GIA Phase A stopped before the FIA grid. GIA is now the `T=1` special case of UIA/FedAvg. |
| 5 | PASS | UIA/FedAvg is the main setting. T-robustness and `B_eff` prediction validation completed without Stage 2/FIA. |
| 6 | SUPERSEDED | Round-dependence spectrum completed; the old GMA/UIA nightly run is superseded by M7. 4-bit GMA is closed after fp32 failure. |
| 7 | EXECUTED; CONTROL INVALID | no-CPA failed variable control. Dependent IUT, normalized metrics and success thresholds/counts are withdrawn; independent CPA/UIA measurements remain. |
| UIA fp32 successor | REPORT READY | n=16/T=10, all80 images, LPIPS=0.45459726, source recovery=79/80; protocol_difference_review. Runner complete; queue exit-code status inconsistency documented. |

<!-- M7_FOLLOWUP_RESULTS_BEGIN -->
## M7 follow-up result summary

- Tasks 1/2/3/5: **COMPLETE**; Task 4 `m7_uia_4bit_B16`: **COMPLETE**.
- No-CPA-referenced normalized ratios (including the earlier 68.2878% and 76.6870%) are **WITHDRAWN**.
- Shuffled means: fp32pure36 **0.808424**,4bit40 **0.798902**,UIAfp3280 **0.816526**; all20 seeds and per-pair distributions saved.
- The no-CPA-derived LPIPS cutoff, all derived success counts and the majority-success interpretation are **WITHDRAWN**.
- Canonical fp32 Spearman **+0.031**,4bit **-0.591**. Prediction **81/96(MAE0.219)**; full T1 **39/48**,core per-row T1 **30/36**.
- n32:21.2 is the five-batch prediction mean; matched batch0 predicts24 across runs. Quantizer seeds differ, so this is not identical-input reproduction.
- Pure36 fp32 reporting resolves the prior n=36 issue. Old UIA queue failed/complete discrepancy remains documented below.
- Reproduction and deliverable links: [consolidated review](docs/EXPERIMENT_RESULTS_20260909.md).
<!-- M7_FOLLOWUP_RESULTS_END -->

## Confirmed follow-up scope (2026-09-09)

- Prediction validation canonical result: **81/96 exact, MAE0.219**.
- UIA fp32 n16/T10 all80: **LPIPS0.4546 / PSNR22.406 / SSIM0.3427**, embedding |cos|>0.9 **79/80**, no oracle fills.
- GIA fp32 reference is **0.4465 over36 images**, recomputed after excluding oracle supplements global22,24,32,36. The fp32 n=36 issue is **CLOSED**.
- No-CPA fp32/4bit are **INVALID**; their similar means cannot establish control validity.
- UIA queue **failed** conflicts with80-image completed/verified outputs. Suspected missing exit-code handling error; original queue state/logs are preserved.
- New run tags: `m7_normalized_metrics`, `m7_shuffle_control`, `m7_perimage_distribution`, `m7_uia_4bit_B16`, `m7_consistency_audit`.
- The locked historical success definitions are archived. No-CPA-derived rank and LPIPS-cutoff counts are invalid under spec section 0.10.
- Tasks1/2/3/5 completed; Task4 uses saved updates only. Current Task4 state is in the result summary above. Optional(A)/(B) remain deferred.

## Prior Task (superseded by UIA main setting)

- Direction changed from the GIA Phase-A grid to UIA/FedAvg. Do not run the old GIA grid unless explicitly re-approved.
- UIA Stage 1 is spectrum-only: VGG-16 conv is frozen, `Z=f_conv(X)` is computed once/cached, and only FC layers are locally trained on cached `Z`.
- UIA observation is `Delta theta / lr = sum_t grad_t(W_0) = B_T Z` for `classifier.0.weight`; `T=1` must match the existing GIA observation numerically.
- Current UIA grid is `n in {16,32}`, `T in {1,2,5,10}`, 4-bit stochastic per-row, `run_tag=m5_uia_spectrum`.
- Measurements: SVD rank, `||b_i||` CV vs GIA `||a_i||` CV, `lam_deb/mp_edge`, predicted `B_eff(T)`, and `Spearman(final loss, ||b_i||)`.
- Prohibited in this phase: ICA, Stage 2 solver, FIA, `--refine-A`, and GIA grid execution.
- Closed by current scope: the GIA published-range B=32~1024 grid/table is dropped from the four-page paper.
- Stage 1 default ICA backend remains CPA-style soft-penalty unmixing for GIA/M3 rows, with debiased whitening + MP truncation kept in front.
- Stage 1 reconstructs `A_hat` by least squares from observed `Y_hat` and recovered `S_hat`: `A_hat = Y_hat @ S_hat.T @ inv(S_hat @ S_hat.T)`.
- Stage 2 image-gain pursuit is closed as a negative result; do not use hinge-only `--refine-A`.
- `git_sha` fallback reads `.git/HEAD`; latest rows log `437013a234e92e882fb6744a64995b0ab812841c`.
- `git_sha` fallback reads `.git/HEAD`; latest rows log `437013a234e92e882fb6744a64995b0ab812841c`.
- One `--refine-A` diagnostic was run as `v8_refine_A`; it is not adopted because it worsens retained source cosine. Hinge-only `--refine-A` is now closed as a negative diagnostic and is not a Milestone-4 grid axis.
- Quantization now exposes `quantization.scale_granularity` in `{per_row, per_tensor, per_bucket}` plus `quantization.bucket_size`; cache keys include the granularity.
- Large-B victim collection now supports `data.microbatch_size`; non-distinct label selection now samples dataset indices without enforcing unique labels.
- Reduced Milestone 4 Phase A/B plan remains in `quantization_aware_cpa/results/v10_m4_phase_plan/`, but execution is stopped/deferred.

## Current Task

- The authorized successor `m6_uia_fp32_B16_fia` completed at `2026-09-09 12:21:04 KST`: fp32 UIA n16/T10, five independent round0 client updates/80 images, FC-only SGD0.001, fixed CPA+FIA (FIA25000/TV0.1/seed42). Full80 LPIPS=0.4545972638, PSNR=22.4060, SSIM=0.342655, source recovery=79/80, no missing images or oracle fills. Protocol comparison is in `docs/UIA_PUBLISHED_PROTOCOL.md`; frozen config/lock remain unchanged.
- The successor reservation at `2026-09-09 06:15 KST` used queue worker PID `10652` and predecessor PID `14056`; UIA runner PID `2076` started after M7 completed at `09:37:56 KST`. Both experiments have exited. The UIA runner and report mark completion, but `queue_status.json` records `failed` with a blank exit-code value. The queue's exit-code retrieval/check is the strongly suspected cause; actual OS exit code cannot be recovered from these records. All80 CSV rows, five completed batch logs, ledger and seven recorded output hashes were verified. Preserve raw queue state/logs.
- Updated reporting: UIA fp32 LPIPS0.4546 is0.0066 above the upper endpoint of the published cross-batch16?128 CPA+FIA range0.410?0.448: a range-boundary result. The new user instruction explicitly authorizes `m7_uia_4bit_B16` using saved actual updates. Stage2, `--refine-A`, new training and GMA4bit remain prohibited.
- User requested no ongoing experiment tracking. The successor uses an OS process-exit wait and completion-marker checks; no manual metric polling or intermediate intervention is scheduled.

- Historical no-CPA pipeline: execution complete, **scientifically invalid** due to failed variable control. Its control comparison, calibrated thresholds/counts and dependent deliverables are withdrawn. The locked protocol and original records remain archived; they are not a current task.
- Independent CPA, prior-only and shuffled raw metrics remain available with the scope in the invalidation record. No replacement control or threshold has been authorized by this correction.

## Superseded M6 Task

- TASK 1 complete: `m6_round_dependence`, FC-only checkpoints at rounds `{0,5,20,50,100}`, `n=32`, 4-bit stochastic per-row, five held-out measurement batches, Stage-1 spectrum only. Measurement batches are ImageNet `val` and were not used for training; FC checkpoint training used ImageNet `train` batches `0-4`, SGD `lr=1e-3`, train batch size `32`.
- At each checkpoint, UIA/FedAvg `T=10` `Delta theta / lr` dumps were saved for later analysis without rerunning FC training.
- TASK 2 superseded: GMA fp32 did not establish its required baseline. 4-bit GMA is closed because compression and method effects cannot be separated; the UIA fp32 n=16 FIA comparison is deferred while M7 measures CPA's contribution directly.
- Historical launcher: `quantization_aware_cpa/scripts/run_m6_nightly_jobs.ps1`; M7 disables its 4-bit branch and rejects `-RunGma4Bit` immediately.
- Upstream UIA update collection was patched only at collection/protocol level: `collect_updates.py` now supports `--n_samples_list`, `--freeze_conv`, Python 3 `next(iter(...))`, and one `z` embedding entry per client round. Attack/FIA/metric code paths remain unchanged.
- TASK 3 cleanup complete from existing logs: T-split prediction validation, 81/96 source audit, round spectrum decomposition, round0 train/val gap audit, n=32 prediction reconciliation, `docs/QACPA_SPEC.md`, and local Slide 11 PNG were updated. The likely Drive deck (`Quantize and Sparsify Cocktail Party Attack`) was found earlier, but native edit failed with Google Slides `403 PERMISSION_DENIED`; copying the deck also failed because the app connection lacks Drive copy scope.
- M6 prohibitions carried forward: Stage 2 solver, `--refine-A`, and GIA Phase A. M7 additionally closes 4-bit GMA and authorizes FIA only for its requested controls.
- Closed by current scope: the old GIA published-range comparison grid is dropped; UIA remains the main setting.
- Stage 2 image-gain pursuit is closed as a negative result; do not use hinge-only `--refine-A`.

## Latest Results

- Result audit corrected on 2026-09-09: arithmetic/hash verification did not establish no-CPA control validity. The no-CPA-dependent claims are withdrawn; see the consolidated review and invalidation record.
- `m6_uia_fp32_B16_fia`: LPIPS per batch `[0.449673,0.471821,0.446135,0.445456,0.459901]`; full80 `0.45459726`. Source recovery `[16,15,16,16,16]`; the one unrecovered source is global22, cosine=0.071866, LPIPS=0.781329, included in all80. Actual-update factorization relative L2 ranges `6.02e-5..1.26e-4`, within gate. See the consolidated review for queue-state and fp32 scale-granularity metadata caveats.
- **WITHDRAWN:** M7 2x2 no-CPA contrasts, combined IUT p=0.03125 and historical rank-based success counts. Independent CPA and prior-only raw measurements remain archived.
- M7 preregistration locked before all new M7 metrics at `2026-09-08T20:44:15Z` (`2026-09-09 05:44 KST`), SHA-256 `41f402f5449deaadc92f37ec4a76909eafda329d89f11fb7d84b984580251539`. See `docs/M7_CONTROL_PREREGISTRATION.json` and `.lock.json`.
- `m7_recompute_controls` complete: shuffled over 20 seeds gives fp32 LPIPS `0.809049`, 4-bit `0.798902`; full ImageNet val 50,000-image pixelwise mean `0.805476`, gray `0.829199`, uniform noise seed42 `0.856640`. Historical LPIPS diagonal reproduction passed tolerance `1e-5` (max errors `1.31e-6`, `2.80e-6`). Includes 1,600 shuffled pairs, 120 constant comparisons, and summaries by target victim batch.
- M7 FIA background queue launched `2026-09-09 05:49 KST`, initial PID `14056`, via `scripts/run_m7_controls.ps1`, and completed `2026-09-09 09:37:56 KST`. No-CPA fp32 40, no-CPA 4-bit 40 and prior-only 40 all finished 25,000 iterations, followed by the complete report. Both `results/m7_no_cpa_control/execution_status.json` and `summary.json` record completion.
- Inputs verified against all 40 archived float originals. Fixed selected observation rows include four zero vectors in each precision; they remain in the controls. FIA uses signed rows and row normalization only, with no ICA, whitening or sign correction. Existing fp32 `0.4448` is explicitly a 36-CPA + 4-exact-source hybrid, with indices `[22,24,32,36]` flagged.
- Superseded M6 queue PID `15792` and its fp32 GMA child PID `18416` were stopped before the queued 4-bit GMA branch; existing outputs/logs were preserved. The old `-RunGma4Bit` command is now rejected before any experiment starts.

- `m6_round_dependence`: correct disjoint-data run completed in `109.8s` on `NVIDIA GeForce RTX 5060 Ti`. A first run that trained on measurement batches was discarded; the valid run trains on ImageNet `train` and measures on held-out ImageNet `val`.
- Round aggregate (`round`, train loss, measurement loss, CV, predicted `B_eff`, Spearman): `0: 0.7276, 1.1856, 1.102, 21.2, 0.958`; `5: 0.6881, 1.1833, 1.104, 21.2, 0.959`; `20: 0.5818, 1.1768, 1.109, 21.2, 0.960`; `50: 0.4171, 1.1659, 1.118, 20.6, 0.962`; `100: 0.2448, 1.1529, 1.134, 20.2, 0.963`.
- Round spectrum decomposition: mean `sigma2` rises mildly from `1.330e-7` to `1.371e-7`, `mp_edge` from `2.622e-7` to `2.703e-7`, and `lambda_max` from `3.061e-4` to `3.218e-4`; `lambda_max/mp_edge` stays nearly preserved (`1167.2 -> 1190.4`). This is not shrinking-scale cancellation. Because measurement loss changes only `1.1856 -> 1.1529` (`-2.8%`), the learning-stage-insensitivity paragraph is held rather than claimed.
- Round0 train/val gap audit: train loss is a checkpoint probe, not a moving average, and round 0 is measured before any optimizer step. Train and val caches have the same batchwise labels/full label overlap but disjoint image paths, so the log-visible explanation is split/image difficulty; per-train-batch checkpoint losses were not logged without recomputation.
- Per-batch predicted `B_eff` by round: batch 0 `24,24,24,23,23`; batch 1 `25,25,25,25,23`; batch 2 `22,22,22,21,21`; batch 3 `21,21,21,20,20`; batch 4 `14,14,14,14,14`.
- UIA `T=10` dumps: 25 `.pt` files saved under `quantization_aware_cpa/results/m6_round_dependence/uia_T10_dumps/` with total size about `10.3GB`; index file is `uia_T10_dump_index.csv`.
- M6 artifacts: `quantization_aware_cpa/results/m6_round_dependence/summary.csv`, `aggregate_summary.csv`, `source_norms.csv`, `lam_spectrum.csv`, `round_lambda_vectors.csv`, `training_loss.csv`, `summary.json`, `round_dependence_summary.png`, `round_dependence_spectrum_summary.png`, `slide11_premise_validation.png`, plus `quantization_aware_cpa/results/m6_round_spectrum_decomposition/round_spectrum_decomposition.csv`, `round_spectrum_decomposition.png`, `interpretation.md`, and `round0_train_val_gap_report.md`.
- Google Slides note: local `slide11_premise_validation.png` is updated at 16:9 (`2880x1620`) with the upstream B=8/16/32 trend table, the T=1 UIA/GIA bridge, the 81/96 validation count, the 30/36 FedSGD core subgrid count, and the lower-bound threat note. Native update of slide 11 in Drive deck `Quantize and Sparsify Cocktail Party Attack` failed with permission/scope errors, so the source deck is unchanged.
- Upstream 4-bit trend validation: upstream 4-bit CPA/FIA gives LPIPS `0.5340` at `B=16` and `0.5615` at `B=32`; local CPA+dequant `B=8` gives `0.5439`, between them and consistent with degradation as batch size increases. This is trend agreement, not identical-condition equality.
- `m5_uia_T_robustness`: spectrum-only UIA robustness completed for `n={16,32}`, five batches, three seeds, `T={1,2,5,10,20,50}`. No rank exceeded `n`; ranks were `14-16` for `n=16` and `29-32` for `n=32`. Mean CV stayed high (`n=16`: about `1.14-1.19`, `n=32`: about `1.15-1.20`) and Spearman stayed strongly positive (`n=16`: `0.942 -> 0.905`, `n=32`: `0.967 -> 0.941` from `T=1` to `T=50`). This supports the claim that local step count is not a defense under frozen-conv UIA/FedAvg.
- `m5_uia_T_robustness` artifacts: `quantization_aware_cpa/results/m5_uia_T_robustness/summary.csv`, `aggregate_summary.csv`, `lam_spectrum.csv`, `source_norms.csv`, `t1_sanity.csv`, `summary.json`, and `uia_T_robustness_summary.png`.
- `m5_uia_beff_validation`: direct recount from the original `summary.csv` confirms ICA + Hungarian only, 96 cells, no Stage 2/FIA, with predicted `B_eff` exact in `81/96` cells (MAE `0.219`, max error `2`). No alternate-count source was found under `quantization_aware_cpa/results`.
- `m5_beff_validation_T1_split`: core per-row grid (`n={16,32,64}`, `bits={2,3,4,8}`, three seeds) is `T=1 (= FedSGD)` `30/36` exact (MAE `0.194`), `T=10 (= FedAvg)` `32/36` exact (MAE `0.167`), overall `62/72` exact (MAE `0.181`). Including extra 4-bit per-tensor/per-bucket checks, T=1 is `39/48` and T=10 is `42/48`.
- `m5_uia_beff_validation` artifacts: original files `summary.csv`, `source_cosines.csv`, `lam_spectrum.csv`, `summary.json`, and square-aspect `predicted_vs_measured_beff.png`; cleanup outputs are in `quantization_aware_cpa/results/m5_beff_validation_T1_split/` with `t_split_summary_full_96.csv`, `t_split_summary_core_per_row.csv`, `numeric_source_audit.md`, `paper_sentence.md`, and `predicted_vs_measured_beff_T_split_square.png`.
- Published-protocol comparison note added: `codex/docs/UIA_PUBLISHED_PROTOCOL.md`. It records the CPA Appendix B Table 5 alignment: VGG-16/ImageNet, conv frozen, FC-only client updates, batch `16/32/64/128`, SGD `lr=1e-3`, and the remaining local-schedule difference to match before FIA LPIPS comparison.
- `m5_uia_spectrum`: UIA/FedAvg spectrum-only Stage 1 completed in `35.6s` on `NVIDIA GeForce RTX 5060 Ti` at round0. Rank gate passed for every cell: rank is exactly `n` for `n=16` and `n=32` across `T={1,2,5,10}`. `T=1` sanity is exact against round0 GIA: `A_true`, `Y`, quantized `Y_hat`, and `delta` relative errors are all `0.0`.
- UIA predicted `B_eff` at round0: for `n=16`, `T=1/2/5/10` gives `[12,12,12,13]`; for `n=32`, all four T values give `24/32`. The `n=32,T=1` value matches the existing round0 GIA v10 MP pregrid value for B=32, 4-bit stochastic per-row. The apparent m6 `21.2/32` value is a five-batch mean (`24,25,22,21,14`), while the matched m5/m6 seed0 batch0 prediction is `24`.
- UIA `||b_i||` CV remains close to round0 GIA `||a_i||` CV rather than becoming much more uniform: `n=16` CV is `1.127 -> 1.133 -> 1.147 -> 1.141` vs GIA `1.127`; `n=32` CV is `0.997 -> 1.000 -> 1.008 -> 1.018` vs GIA `0.997`.
- `Spearman(final loss, ||b_i||)` is strongly positive: `n=16` `[0.9265,0.9412,0.9412,0.9441]`, `n=32` `[0.9553,0.9564,0.9472,0.9337]` for `T=1/2/5/10`. This does not support the hypothesis that UIA makes leakage much more sample-uniform in this first batch.
- Artifacts: `quantization_aware_cpa/results/m5_uia_spectrum/uia_spectrum_summary.png`, `summary.csv`, `source_norms.csv`, `lam_spectrum.csv`, `t1_sanity.csv`, and `summary.json`.
- `v10_debias_fallback_report`: replacing `A` fallback from no-debias `CPA+dequant` to full-`B` debiased Stage-1 sources improves full-`n=40` LPIPS only from `0.5413` to `0.5370`; recovery rises from `0.0266` to `0.0697`, still far below `A_oracle` `0.4940` / `0.5041`. On the 10 fallback images, `naive_full` is LPIPS `0.7865`, source cosine `0.2098`; `debias_full` is LPIPS `0.7693`, source cosine `0.2935`.
- `v10_source_diagnostics`: oracle-extra weak-source quality is essentially the same as `A_oracle` on the 10 sub-threshold sources: mean source-cosine delta `-0.0017` (`min=-0.0061`, `max=0.0074`). The old oracle-extra image-gain vs fallback-contribution mismatch is therefore mostly selection/threshold/accounting, not an oracle-extra `S` quality failure.
- Batch 2 retained-gain diagnosis: batch 2 retained `CPA+dequant` source cosine mean is `0.8563` with one source near `0.0109`, while the other batches are `0.9958-0.9985`. This explains why retained `A` helps batch 2 (`+0.0443` LPIPS) but not the others.
- `v10_mp_prediction_grid`: MP-only pregrid completed for 28 conditions. Per-row distinct B_eff is B=32 `[14,21,24,25,28]`, B=128 `[33,59,75,95,105]`, B=512 `[56,169,239,324,376]` for bits `[2,3,4,6,8]`. At B=512, corrected non-distinct selection has `384` unique labels and `128` duplicates; mixed B_eff is `[56,162,229,320,378]`, so duplicate labels mildly reduce B_eff at 3/4/6-bit but not monotonically at 8-bit.
- Scale granularity pregrid: per-tensor/per-bucket strongly reduces B_eff. Examples: B=32 bits=3 per-row/per-bucket/per-tensor is `21/7/5`; B=128 bits=4 is `75/45/32`.
- `v8_pre_refine_diagnostics`: retained-source comparison is in `quantization_aware_cpa/results/v8_pre_refine_diagnostics/`. On the 30 MP-retained sources, A_oracle beats A on 28/30; mean retained cosine is A `0.996184` vs A_oracle `0.999036` (`+0.002852`). A has 3 retained sources below `0.99`; A_oracle has 0.
- Retained 30-image check is in `quantization_aware_cpa/results/v8_m3_retained30_image_check/`: fp32 LPIPS `0.4464`, CPA+dequant `0.4631`, A `0.4596`, A_oracle `0.4511`. On the same retained images, A_oracle is better than A.
- Batch LPIPS/gap table is `quantization_aware_cpa/results/v8_pre_refine_diagnostics/batch_lpips_gap_recovery_table.csv`. Gap-vs-recovery correlation across 5 batches is Pearson `-0.305` (`p=0.618`) and Spearman `0.500` (`p=0.391`), so the "larger CPA+dequant-fp32 gap explains recovery" hypothesis is weak at this sample size. CPA+dequant-LPIPS-vs-recovery is negative: Pearson `-0.402`, Spearman `-0.400`.
- `v8_refine_A`: alternating S-step plus hinge-only A-step was executed over 5 batches. It reduced hinge H at every A-step and accepted 25/25 A updates, but worsened retained source cosine on 30/30 sources. Mean retained cosine fell from A `0.996184` to refine-A `0.982803`; minimum fell to `0.922622`. Do not adopt this refine-A path without a new regularized A prior.
- `v9_stage2_saturation_diagnostics`: the full-row `A_oracle - A` LPIPS gap is `0.0473`. The 30 retained images contribute only `0.0063` to the full mean; the 10 CPA+dequant-fallback images contribute `0.0410` (`86.6%` of the `A_oracle - A` gap). This explains the apparent missing 80% of A_oracle's 50.4% recovery as fallback/set accounting, not a reason to continue Stage-2 image tuning.
- Batch 2 context from `v9_stage2_saturation_diagnostics`: it is the only positive A recovery batch because it has the highest retained ratio (`B_eff=7/8`, fallback `1/8`) and retained A improves CPA+dequant by `0.0443` LPIPS on average. Other batches have lower retained ratios and negative retained A gains.
- Full-n image grids for the paper draft are in `quantization_aware_cpa/results/v9_stage2_saturation_diagnostics/figure_original_fp32_4bit_naive_4bit_A_fulln.png` and the loss-sorted companion.
- Current paper-figure assets are staged in `quantization_aware_cpa/results/v9_paper_figures/`: MP-margin scatter, loss-vs-LPIPS scatter with Spearman CI in the legend, the full-n image grid, and the M4 heatmap input schema.
- `v7_m3_selection_effect`: recomputing on the exact 30 global indices selected by non-fallback `A` gives fp32 Spearman `0.0558`, CPA+dequant `-0.0968`, and A `0.0345`. Since CPA+dequant@30 is near zero, the previous A Spearman improvement is a selection effect, not a standalone method effect.
- `v7_m3_fp32_truth_missing`: fp32 missing global indices `[22, 24, 32, 36]` were recovered with exact-source FIA. They were missing because the old fp32 row used `effective_centered_rank(victim.Y)`, which reduced batch 2 by 1 source, batch 3 by 1 source, and batch 4 by 2 sources.
- `v7_m3_fallback_5batch` is an archived hybrid aggregate. Current fp32 reporting excludes its four oracle fills and uses pure36 **LPIPS0.4465, Spearman+0.031**; 4-bit all40 is **LPIPS0.5439, Spearman-0.591**. The earlier hybrid correlation is not a second current estimate. Historical Stage2 rows remain archived negative results.
- Full-n batch recovery rates are: batch 0 `-0.090`, batch 1 `-0.084`, batch 2 `0.456`, batch 3 `-0.056`, batch 4 `-0.030`. The earlier batch 2/4 high recovery impression does not reproduce under the full-`n=40` denominator.
- Batch 2 non-fallback A mean `LPIPS=0.4382` is the retained 7-source set with global indices `[16, 17, 18, 19, 20, 21, 23]`; the missing/fallback source is global `22`.
- `v6_m3_spearman_5batch`: aggregated M3 image report over batch 0-4 is in `quantization_aware_cpa/results/v6_m3_spearman_5batch/`. All-image means: fp32 `n=36, LPIPS=0.4465, PSNR=21.912, SSIM=0.3467, Spearman=0.0309`; CPA+dequant `n=40, LPIPS=0.5439, PSNR=21.536, SSIM=0.3070, Spearman=-0.5906`; A `n=30, LPIPS=0.4596, PSNR=21.776, SSIM=0.3480, Spearman=0.0345`; A_oracle `n=40, LPIPS=0.4940, PSNR=21.717, SSIM=0.3384, Spearman=-0.4749`.
- Batch 4 completed in `quantization_aware_cpa/results/v6_m3_fia_4bit_batch4/`: fp32 `0.4843`, CPA+dequant `0.5960`, A `0.5018`, A_oracle `0.5706` mean LPIPS. Batch 4 A_oracle Spearman is `-0.881`.
- Fair same-image check for the original v5 retained set is `quantization_aware_cpa/results/v6_m3_fair_common_retained_summary.csv`. On the common 12 images, A_oracle beats A (`LPIPS 0.4566` vs `0.4656`), so the earlier full-row `A < A_oracle` observation was a sample-set artifact.
- Oracle-extra rank-split FIA is in `quantization_aware_cpa/results/v6_diag_rank_split_fia/`: selected `34/40` sources, mean LPIPS `0.4644`, PSNR `21.803`, SSIM `0.3449`, Spearman `-0.1505`. This is only a tiny image-level gain over the old A row (`0.4656`), so rank-split is currently a weak upper bound rather than a strong next-step signal.
- Latest local tests: `C:\miniconda\envs\DLG\python.exe -m pytest` passes (`5 passed, 1 skipped`).
- `v5_A_lsrescale_cpaica`: actual estimated-A row improved from old `[4/8, 3/8, 5/8, 6/8, 3/8]` to `[6/8, 6/8, 7/8, 6/8, 5/8]`; over `B_eff` it is `[6/6, 6/6, 7/7, 6/6, 5/5]`.
- LS scaling barely changes rank reconstruction (`pre_ls` to `post_ls` rel-L2 stays about `[0.558, 0.549, 0.477, 0.540, 0.536]`) but fixes absolute cell scale: actual-A `H_init` is `[19.1, 59.9, 10.8, 17.6, 32.1]` instead of the old Stage-2 `H` values up to `4.08e5`.
- `v5_A_oracle_guard_cpaica`: oracle row remains `[7/8, 8/8, 7/8, 6/8, 5/8]`; sub-threshold recovered counts are `@0.9=[1,2,0,0,0]`, `@0.7=[2,2,1,1,1]`, `@0.5=[2,2,1,1,3]`.
- `v5_bplus_truncated_lsrescale_cpaica`: `[6/8, 6/8, 7/8, 6/8, 5/8]`; sub-threshold recovery remains zero at `0.5/0.7/0.9`.
- `v5_bplus_fullB_lsrescale_cpaica`: `[7/8, 6/8, 7/8, 6/8, 5/8]`; sub-threshold counts are `@0.9=[1,0,0,0,0]`, `@0.7=[1,0,0,0,0]`, `@0.5=[1,0,0,1,0]`. Full-B collapse is not a CPA-ICA result.
- `v5_m3_fia_4bit`: upstream-style FIA on the same 16 images. Mean metrics: fp32 `LPIPS=0.4448, PSNR=21.619, SSIM=0.338`; 4-bit CPA+dequant `0.5423, 21.340, 0.304`; 4-bit A `0.4656, 21.387, 0.340` on 12 retained images; 4-bit A_oracle `0.4718, 21.531, 0.330`. Spearman(loss, LPIPS): fp32 `0.006`, CPA+dequant `-0.656`, A `-0.280`, A_oracle `-0.444`.
- M3 artifacts: `results/v5_m3_fia_4bit/summary.json`, method-level `per_image_metrics.csv`, PNG dumps, `figure_original_fp32_4bit_naive_4bit_A_oracle.png`, and `figure_original_fp32_4bit_naive_4bit_A_oracle_loss_sorted.png`.
- `v5_diag_rank_split`: Stage 2 with `[A_hat_keep | A_true discarded]` confirms rank loss. Recovered over B is `[7/8, 8/8, 7/8, 7/8, 5/8]`; sub-threshold recovered counts are `@0.9=[1,2,0,1,0]`, `@0.7=[2,2,1,1,1]`, `@0.5=[2,2,1,1,3]`.
- `v5_rank_split_residual`: non-oracle residual short-ICA implementation preserves retained sources but does not recover sub-threshold sources. Recovered over B is `[6/8, 6/8, 7/8, 6/8, 5/8]`; sub-threshold counts are `@0.9=[0,0,0,0,0]`, `@0.7=[0,0,0,0,0]`, `@0.5=[0,1,0,0,0]`. Extra source cosines from residual ICA are only about `0.0046-0.027`, so extra-column estimation, not Stage 2, is the current bottleneck.
- New output files: `results/v5_4bit_table_lsrescale_cpaica.csv`, `results/v5_subthreshold_cosines_lsrescale_cpaica.csv`, `results/v5_subthreshold_threshold_counts_lsrescale_cpaica.csv`, `results/v5_A_oracle_minus_naive_componentwise_cpaica.csv`, `results/v5_subthreshold_histogram_lsrescale_cpaica.png`.
- Upstream fp32 image dump: `official_cpa/exp/imagenet/vgg16/attack/cp_direct/nodef/fp32_B32_full/fp32_image_dump/` contains 16 original/reconstruction PNG pairs, a contact sheet, and `fp32_B32_per_image_metrics.csv` with mean LPIPS `0.4660175`.
- `v4_cmp_sweep_cpaica`: `c_mp=1.0` no longer collapses under CPA ICA (`6/8, 6/8, 7/8`), but keeps weak near-noise axes with worst cosines `[0.0104, 0.0075, 0.0160]`. `c_mp=3` and `6` plateau at `[6/8, 6/8, 7/8]`; `c_mp=12` over-truncates batch 2 to `6/8`.
- `v4_lamR_tune_cpaica`: held-out batch 1 A row is unchanged for `lam_R in {0,30,100,300,1000}`: recovered `3/6 (3/8)`, `H=4.08e5 -> 4.08e5`, consistency `0.93135 -> 0.93135`.
- `v4_debias_cpaica`: Stage-1 debias-only row over B is `[6/8, 6/8, 7/8, 6/8, 5/8]`; over `B_eff` it is `[6/6, 6/6, 7/7, 6/6, 5/5]`.
- `v4_naive_cpaica`: no-debias row over B is `[6/8, 6/8, 6/8, 6/8, 5/8]`.
- `v4_bplus_truncated_cpaica`: bplus truncated row over B is `[6/8, 6/8, 7/8, 6/8, 5/8]`; sub-threshold recovery is zero at both cosine thresholds.
- `v4_bplus_fullB_cpaica`: full-B bplus is no longer the old FastICA collapse under CPA ICA; over B is `[7/8, 6/8, 7/8, 6/8, 5/8]`, with sub-threshold recovered `@0.9=[1,0,0,0,0]`, `@0.5=[1,0,0,1,0]`.
- `v4_A_oracle_cpaica`: A_oracle over B is `[7/8, 8/8, 7/8, 6/8, 5/8]`, over `B_eff` is `[6/6, 6/6, 7/7, 6/6, 5/5]`, sub-threshold `@0.9=[1,2,0,0,0]`, `@0.5=[2,2,1,1,3]`.
- `v4_A_cpaica`: actual estimated-A row over B is `[4/8, 3/8, 5/8, 6/8, 3/8]`, over `B_eff` is `[4/6, 3/6, 5/7, 6/6, 3/5]`. This is the current `A_hat` estimation bottleneck.
- Summary CSVs: `results/v4_cmp_sweep_cpaica.csv`, `results/v4_4bit_table_cpaica.csv`, and `results/v4_subthreshold_cosines_cpaica.csv`.
- `gma` was not generated by this embedding-stage harness; it requires the upstream/full-model image-gradient path and is not represented by source-cosine rows here.
- `v3_mptrunc`: `B_eff=[7, 6, 7]`, recovered over `B_eff=[6/7, 6/6, 7/7]`, recovered over `B=[6/8, 6/8, 7/8]`.
- `diag_deflation`: no truncation, recovered over `B=[6/8, 4/8, 7/8]`, `fastica_n_iter=[11, 12, 8]`.
- CUDA check: `NVIDIA GeForce RTX 5060 Ti`, capability `(12, 0)`.
- `v3_cmp_sweep`: `c_mp=1.0` keeps all 8 and gives `0/8` in all three batches. `c_mp>=3` gives `6/6, 6/6, 7/7` except `c_mp=12` gives `6/6, 6/6, 6/6`.
- `v3_feas_decomp`: A_true full feasibility is `1.0` for all three batches; true-kept vs A_hat gaps are `[0.508, 0.400, 0.562]`.
- `v3_A_oracle`: recovered over B for 5 batches is `[7/8, 6/8, 7/8, 6/8, 5/8]`. Hinge-only Stage 2 does not restore the low-gradient sources from pinv initialization.
- `v3_stage2_noop_check`: hinge-only Stage 2 max source-cosine change per batch is at most `0.00224`, so it is effectively a no-op from the pinv initialization.
- `v3_axis_extent`: coordinate extent vs `1/||a_i||` Spearman correlations are `[0.881, 0.929, 1.000, 0.976, 0.952]`.
- `v3_l1_tune`: old dimensional-penalty L1 selected `lam_R=3e-5`, but this run is superseded.
- `v3_A_oracle_l1`: old penalty form recovered over B `[7/8, 7/8, 3/8, 7/8, 5/8]` and over-shrank batch 2; superseded.
- `v3_fista_l1_tune`: held-out batch 1 selected fixed dimensionless `lam_R=300` under the two-phase FISTA+guard solver.
- `v3_A_oracle_dimless_l1`: recovered over B `[7/8, 8/8, 7/8, 6/8, 5/8]`; recovered over B_eff is `[6/6, 6/6, 7/7, 6/6, 5/5]`; `sub_threshold_recovered=[1,2,0,0,0]`.
- `v3_bplus_dimless_l1`: recovered over B `[6/8, 6/8, 7/8, 6/8, 5/8]`; within retained `B_eff`, bplus is recovered on all five batches; `sub_threshold_recovered=0` for every batch.
- Batch 2 is improved by dimensionless Phase 1 but not fully recovered: source 6 reaches cosine `0.7650`, so the batch remains `7/8`.
- `v3_dimless_check_6bit`: same `lam_R=300` with 6-bit, no retune, recovered over B `[8/8, 8/8, 7/8, 7/8, 7/8]`; recovered over B_eff is `[7/7, 8/8, 7/7, 7/7, 5/5]`; `sub_threshold_recovered@0.9=[1,0,0,0,2]`, `@0.5=[1,0,1,0,3]`.
- `v3_bplus_fullB`: bplus with no MP truncation and full B=8 recovered over B `[1/8, 0/8, 1/8, 0/8, 0/8]`; `sub_threshold_recovered@0.9=[0,0,0,0,0]`, `@0.5=[1,0,0,0,0]`.
- 4-bit posthoc sub-threshold counts for `v3_A_oracle_dimless_l1`: `@0.9=[1,2,0,0,0]`, `@0.5=[2,2,1,1,3]`.
- Upstream CPA objective: optimize row-normalized `W_hat` on whitened gradient rows with loss `decor*mean(exp(T*|cos(W_i,W_j)|)-1) + nv*mean(min(||relu(-S)||,||relu(S)||)) + l1*mean(|S|) + tv*TV(S)`; code sets neg-entropy term to zero after computing it. Optimizer is Adam/SGD via `get_opt`, README ImageNet uses Adam `lr=1e-3`.
- Upstream fp32 B=32 reproduction: embedding CS mean `0.94426`; CPA+FIA on 16 sampled images gives mean LPIPS `0.46602`, PSNR `24.0571`, SSIM `0.3678`, matching/improving paper Table 2 CPA+FIA B=32 LPIPS `0.483`.

## Void Runs

- Initial rows produced before the whitening/debias fixes are tagged `run_tag=void_prewhitening_bug` and must not be mixed into paper numbers.
