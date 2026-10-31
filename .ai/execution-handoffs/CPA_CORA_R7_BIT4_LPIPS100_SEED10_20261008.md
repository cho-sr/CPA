# CoRA 4-bit endpoint LPIPS matrix with seed control

TASK_ID: CPA_CORA_R7_BIT4_LPIPS100_SEED10_20261008
PLAN_VERSION: 1
STATUS: COMPLETED
ASSIGNED_ROLE: CODEX_EXECUTOR
CLAIMED_BY: current session 01a113de-90d5-7f81-acc3-91e6b618d2a7 (explicit user direct-execution override)
REQUIRED_MODEL: USER_AUTHORIZED_CURRENT_SESSION (no model change)
REQUIRED_REASONING_EFFORT: CURRENT_RUNTIME
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_REQUEST_2026-10-08 -- compare all 100 final candidate reconstructions using a 100x100 pairwise LPIPS matrix; run one fixed candidate ten times with different optimization seeds and compare D_seed with D_candidate. User explicitly selected changing BOTH CPA and FIA seeds in the clarification response.
DELIVERY_STATUS: NOT_DELIVERED_EXISTING_EXECUTOR_EXITED -- actual executor_bridge --send refused because bound CLI PID 3140 is no longer running. No research worker or LPIPS computation was launched.

## Role and prior work

The session remains CODEX_PLANNER under .codex/INSTRUCTIONS.md and .ai/THREE_CLI_ROLES.md. Experiment authorization is already provided; do not ask for it again. The currently bound existing Executor PID 3140 was found exited on 2026-10-08. Do not spawn a replacement agent or rebind to an arbitrary session. A user-directed executor/role correction can supersede this delivery constraint.

The preceding package task is complete: its ZIP, .sha256 and .build.json exist under Desktop/fedavg_cora_transfer_20261006. Its old READY handoff is preserved as history; do not restart packaging or modify that ZIP.

## Scope and data

Scope is the existing round007, client00/input_batch00, bit04, B16 result, for which all 100 raw endpoints survive. Do not silently label it the newer five-batch server experiment. No endpoint regeneration for rounds 1, 3 or 5 is authorized in this task.

Resolve BASE as Path.home()/Desktop/fedavg_cora_transfer_20261006/fedavg_cora_transfer_20261006.

- RUN: BASE/imagenet-fedavg/runs/cora_full_r1357_c3_b16_bits84_c100_full25k.
- ENDPOINTS: BASE/cora_before_cleanup/public/round007/client00/bit04/candidate000..099/rec.pt.
- SOURCES: the corresponding candidate directories contain sources.pt.
- Existing result metadata: RUN/public/round007/client00/bit04/candidateNNN/result.json, fia_result.json.
- Condition manifest: RUN/public/round007/client00/bit04/manifest.json.
- Fixed matching: RUN/evaluation/round007/client00/bit04/per_image_metrics.csv, source_index -> estimated_source_index for each candidate; 16 one-to-one matches per candidate.
- Model checkpoint: BASE/imagenet-fedavg/runs/imagenet_pretrained_full_reset_fc_E5_R20/rounds/round_007.pt.
- Model SHA256: 3a78c609b913a8c85503fbcf9ca0ccb5c5b7107eaf37fdab989df151489c8142.
- Original public worker: BASE/imagenet-fedavg/cora_attack.py and cora_common.py; use construct_fia, RoundFeatureModel, runtime and the original upstream FIA implementation.
- Local environment: C:/miniconda/envs/DLG/python.exe; observed torch 2.13.0+cu132, RTX 5060 Ti 16GB. torchmetrics, torchvision, lpips, scipy and matplotlib are installed. VGG16 weights are already cached.

## Fixed design before execution

The user explicitly selected changing BOTH CPA and FIA optimization seeds. The control estimates complete CPA-plus-FIA optimization variability at one fixed quantization candidate. The FIA target is allowed to vary as the result of the CPA rerun; do not freeze sources.pt or reuse it as the control's CPA result.

Control candidate: 001, the first random candidate by ID. This deterministic choice does not select a candidate by measured LPIPS or reconstruction quality. Candidate000 is a special center and is not used as the sole stochastic control.

Freeze the candidate001 input matrix exactly. Reconstruct it once from the recorded packet with the original sampler, sample_seed=1001237 and chunk_rows=32. Verify its tensor hash against sample.json: b402e70cd71f4ff148c094e59b5d547fed508ac375ce82d3853b8e9e0417c349; verify code/scale consistency. Do not change the quantization sampling seed. Freeze the model checkpoint, packet, batch, precision, optimizers, schedules, losses, horizons and all other settings.

For trial t in [0,1,2,3,4,5,6,7,8,9], use CPA seed=t and FIA seed=42+t. Every trial performs CPA 25000 iterations from scratch, then FIA 25000 iterations from scratch using abs(that trial's CPA sources). Both optimization seeds vary together; this is ten paired-seed trials, not a 10x10 factorial. Trial0 reproduces the historical CPA0/FIA42 seed pair locally and is included among the ten new runs. Historical saved endpoints do not substitute for any of the ten new trials. Seed IDs, stage, exact matrix/model hashes and horizon must enter checkpoint bindings to prevent accidental reuse across trials.

CPA settings copied from the recorded manifest: 25000 iterations, Adam lr=0.001, scheduler=none, decor=5.3, T=7.7, ne=1.0, nv=0.13, l1=5.0, tv=0.1. Preserve the exact historical upstream implementation, including its actual loss behavior; do not substitute the separate negentropy_enabled variant.

FIA settings copied from the recorded manifest: Adam, lr=0.1, cosine schedule across the full 25000 iterations, tv=0.1, fi=1.0, no labels, gm=0. All 16 images remain in the optimization batch. Preserve runtime and normalization behavior from the original implementation. No tuning, early stopping, top-k screening or additional seeds. Verify the original upstream source against worker00.json code hashes, translating only server paths to the corresponding local cora-reference package.

## LPIPS definition and image correspondence

Use calibrated VGG LPIPS, eval mode, FP32, RGB 224x224, with stored [0,1] reconstruction pixels mapped to [-1,1]. Record exact library versions and learned/backbone weight hashes. Prefer the installed TorchMetrics VGG LPIPS definition already used by cora_report.py. Do not resize, crop, augment or match by minimizing pairwise LPIPS.

For each existing candidate c, reorder its 16 reconstructed images with the saved estimated_source_index permutation sorted by source_index. The matching is from already-completed private evaluation and is used only for correspondence in this posthoc analysis. Do not use private truth for optimization or choosing the control candidate.

For each source image i, calculate a symmetric 100x100 LPIPS matrix. Store the complete [16,100,100] array as well as the requested mean matrix:

    D_candidate[c,d] = mean_i LPIPS(x[c,i], x[d,i]).

For the ten control endpoints, recompute the original Hungarian absolute-source-cosine match for each trial's new CPA sources against RUN/private/round007/client00/truth.pt Z_evaluation_only. The CPA seed can permute recovered source rows; do NOT reuse candidate001's historical permutation for the new trials. Follow cora_report.py/match_sources exactly, and keep this matching strictly after endpoints are frozen. Record each permutation, matched cosine, and any source-matching ambiguity. Use the same assigned source identity for the corresponding FIA images. Store [16,10,10] and:

    D_seed[s,t] = mean_i LPIPS(x[001,s,i], x[001,t,i]).

Evaluate upper triangles only and mirror for output; check self distances before setting display diagonals to zero. If caching normalized LPIPS feature maps accelerates computation, verify cached distances against the canonical metric on several distinct pairs first. Do not call the metric with unintended stateful accumulation. Chunk to bounded CPU/GPU memory.

The full candidate matrix includes all IDs 0..99. Summaries use 4950 unique off-diagonal candidate pairs and 45 unique seed pairs. Also report the predeclared random-only 99-candidate subset separately because center000 has different sampling semantics; this does not replace the full matrix.

## Comparison and interpretation

Report medians, means, standard deviations, quantiles, ECDF/histogram overlap, and median(D_candidate)/median(D_seed), both for batch-mean matrices and separately for each of the 16 images. Include the candidate001-to-other-candidates distances as an anchored comparison to its own seed control.

Pair distances share endpoints and are not independent observations. Do not report a naive n=4950 versus n=45 significance test or bootstrap individual pairs as independent. A single controlled candidate cannot establish seed variability for all candidates. Do not turn an arbitrary LPIPS cutoff into a basin label.

Measure LPIPS between rerun trial0 (CPA0/FIA42) and the stored candidate001 CPA0/FIA42 endpoint after aligning both by the same source identity, separately from both distributions. Record software, GPU and implementation differences; if this same-seed discrepancy is material relative to either distribution, mark the candidate-vs-seed comparison as environmentally confounded rather than attributing all separation to candidate identity.

Interpret distances comparable to seed variation as no clear candidate-specific separation beyond this control; substantially larger distances support candidate-dependent endpoint changes. Neither outcome alone proves distinct optimization basins because candidate FIA targets differ. No loss-barrier or extra optimization experiment is authorized here.

## Implementation, budget, and outputs

Write new analysis code/config/results only in a new versioned directory under BASE/analysis_exports, suggested cora_round7_bit4_lpips100_seed10_20261008. Preserve all source, historical endpoints, original code, manifests and the existing server ZIP. An output path collision must resume its authenticated matching task state or choose a clearly versioned new path; do not overwrite unrelated results.

Verify all endpoint/source input hashes against recorded result metadata, checkpoint hash, tensor shape/range/finiteness, and one-to-one correspondence. File-hash inequality alone does not prove different tensor values. Check LPIPS matrices are finite, symmetric and correctly indexed with self distances near zero. Use isolated seed directories and authenticated resumable optimizer/RNG states. Do not reuse a checkpoint from another seed or target. A failed integrity or nonfinite check stops dependent work and reports the cause without automatic restart.

Run at most one control optimization at a time on the available GPU unless resource evidence supports safe execution without affecting scientific conditions. Check for an existing task process and available memory before launch. No extra pilot runs. Historical server CPA-plus-FIA time for candidate001 was 1370 seconds; ten runs represent about 3.8 hours on that historical machine, not a local runtime guarantee. Estimate local time from the first actual scheduled trial's progress. The user-authorized optimization budget is exactly ten full CPA-plus-FIA runs (250000 CPA + 250000 FIA steps); matrix evaluation is additional read-only inference. Preserve at least 8 GiB free disk as in the original worker.

Deliver:
- run_manifest.json with input/code/library/weight hashes, control candidate, seed list, configuration and actual execution evidence;
- candidate_pairwise_lpips.npy and labeled .csv (100x100), plus per-image matrices;
- seed_pairwise_lpips.npy and labeled .csv (10x10), plus per-image matrices;
- all ten final control endpoints and per-seed completion metadata;
- candidate heatmap, seed heatmap and distribution comparison plots;
- comparison.json and a concise Korean result summary with limitations and same-seed reproducibility distance;
- honest status if incomplete, with no placeholder/fabricated seed-control distances.

The Executor must complete required code checks, launch durably, register the single non-model process-exit notification described in WORK_MODE.md, and end its turn after one launch confirmation. No repeated model-driven monitoring or extra experimental arms. On completion perform the authorized comparison/report and deliver the result once.

## Explicit direct-execution override -- 2026-10-08

User: "?? ? ??". This authorizes the current session to implement and execute this concrete task directly, superseding the Luna-only delivery restriction for this task. No model identity change is claimed. The prior failed-delivery field is historical; current delivery mode is DIRECT_CURRENT_SESSION.

## Actual direct launch

Parent PID 18188, creation epoch ms 1791447203467. One immediate launch check confirmed status RUNNING, stage candidates, worker PID 3588. Ten CPA+FIA trials and the final evaluation/report are scheduled in the same durable pipeline. Launch receipt: .ai/execution-handoffs/CPA_CORA_R7_BIT4_LPIPS100_SEED10_20261008.execution.json. No completion is claimed.


## Actual failure and one-time completion review -- 2026-10-08

STATUS: FAILED. Finished at 2026-10-08T08:13:42.209158+00:00 (17:13:42 KST).
The candidates stage exited with code 1 at the LPIPS Gram numerical-consistency
check (run_study.py:249; RuntimeError: LPIPS Gram numerical error).
The log does not include the failing numerical values; the precise numerical
cause is unconfirmed. All 100 historical endpoints were loaded and aligned, but
no complete candidate LPIPS matrix was produced. CPA+FIA control trials started:
0 of 10. No comparison.json or final plots exist; no basin conclusion is available.
Evidence: output status.json and candidates.log. The user completion callback
explicitly prohibits restarts, additional seeds, and polling. This review updated
records only; it did not modify the driver or restart any experiment.


## Explicit user-authorized retry -- 2026-10-08, attempt 2

The user explicitly requested rerun after the failure report. The existing
task-specific direct-execution override remains in force; SESSION_ROLE remains
CODEX_PLANNER. Exactly the same round7/client0/bit4, 100 endpoints, candidate001
and ten paired CPA/FIA seeds, 25000 steps per stage are retained.
Replace cancellation-prone FP32 Gram distances with direct normalized-feature
squared differences, the installed TorchMetrics LPIPS calibration layers and
spatial averaging. Existing tolerances and canonical-forward checks are retained;
check failures now include actual numerical values. Authenticated aligned input
cache is reused. No control optimization had started in attempt 1.
The complete original driver/manifest/status/logs are preserved in
analysis_exports/cora_round7_bit4_lpips100_seed10_20261008/attempt01_failed.
One durable pipeline and one terminal completion callback; no model polling.


## Actual retry launch -- attempt 2

Launched 2026-10-08 17:52:55 KST (08:52:55 UTC). Parent PID 20596,
creation epoch ms 1791449575369, candidate worker PID 18084. One initial
confirmation after 20 seconds found RUNNING and source images 1 through 5 of 16
completed, including direct-versus-canonical LPIPS checks. The original failing
numerical stage has been passed for those images. This is progress, not a final
100x100 matrix or completed seed-control result. The same durable process will
run the ten CPA+FIA controls and final comparison, then notify this thread once.
No model-based progress polling is scheduled.


## Final result -- 2026-10-08T14:53:25.313525+00:00

STATUS: COMPLETED. Candidate LPIPS matrix: 100x100 over 16 matched images,
4,950 unique off-diagonal pairs. Seed control: 10 CPA+FIA paired-seed runs
(CPA 0..9, FIA 42..51), 45 unique pairs. Batch-mean LPIPS medians: candidate
0.648113, seed 0.668309;
ratio 0.969780. Historical
versus local same-seed replay mean: 0.672943,
1.006933x the
seed-pair median. This replay distance is comparable to seed variability, so
environmental differences confound attributing distances to candidate identity.
No basin conclusion is established by LPIPS alone. See comparison.json and plots.
