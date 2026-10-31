# H1 actual-update cell screen: remove the FIA target term

TASK_ID: CPA_SIGNED_E2E_H1_NOFIA_20260916
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: IMPLEMENT_VALIDATE_AND_RUN_SINGLE_FACTOR_ABLATION
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_GO_FOR_PROPOSED_H1_FIA_1_VS_0_COMPARISON_20260916
REVIEW_REFERENCE: USER_ACCEPTED_PLANNER_RESULT_ANALYSIS_AND_SINGLE_FACTOR_PROPOSAL
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_E2E_H1_NOFIA_20260916/report.md
EXECUTION_SUMMARY_PATH: .ai/execution-results/CPA_SIGNED_E2E_H1_NOFIA_20260916/execution_summary.json
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_e2e_h1_nofia_B16_T10_20260916
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916

## Authorization and scope

The user requested analysis of CPA_SIGNED_E2E_CELL_SCREEN_20260916. Planner proposed: keep H1, initial image, TV, cell weight and budget unchanged, and compare FIA weight 1 versus 0 to isolate pressure from the fixed CPA target. The user replied `ㄱㄱ` (go ahead). This explicitly authorizes the bounded implementation, validation, one new 100-step arm and final evaluation below. The prior general authorization to resolve ambiguities experimentally remains context, but is not an unlimited search budget.

Reuse the completed B_H1_high as the lambda_F=1 control; do not retrain it. Run one new arm C_H1_high_noFIA with lambda_F=0. This is a fresh continuation from the SAME naive x0, not a continuation from B_H1_high. Implementing utilities alone is not completion: execute the authorized comparison and report. No new Claude discussion or approval is needed to carry out this user-accepted proposal. Do not claim Claude separately reviewed this exact follow-up.

Planner analysis: .ai/analyses/CPA_SIGNED_E2E_CELL_SCREEN_20260916/analysis_ko.md. It documents the public H0 feasibility contradiction, H1 uncertainty, bias-heavy loss, gradient conflict, and weak fixed CPA target. Those are motivations, not permission to change label search, tensor weights or priors in this ablation.

Designated Executor is the existing user-initialized Luna/xhigh thread 01a0a799-f1ce-7b31-8c46-b4ddef40aeab, observed PID14236, process creation1789518279030ms. Reverify thread/runtime/explicit role and delivery hash before claim. This task-specific designated runtime takes precedence over historical Astra/high examples in generic collaboration docs; do not switch models or start another CLI. Existing Planner return thread is 01a0a799-d62c-7ab0-9c2a-628acf6d2c95, observed PID24972, creation1789518271364ms. Reverify before the one final notice.

## Fixed inputs and single changed variable

Previous completed public root: quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916.
Previous private root: quantization_aware_cpa/results_private/signed_e2e_cell_screen_B16_T10_20260916.
Use the same signed observation at results/signed_int4_baseline_B16_T10/observation, the same naive/candidate.pt x0 and naive/sources.pt target_z, and the same public VGG16 checkpoint at C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth. Authenticate actual paths/hashes against the previous registration/integrity/summary before any real optimization. The prefixes above are under quantization_aware_cpa unless already absolute.

Read H1 from the authenticated previous label_hypotheses.json and use exactly [507,655,452,549,231,657,160,761,975,389,658,761,621,124,969,290] in the existing image order. Keep it fixed. Do not recompute assignment from the changed images, use private labels, or treat this vector as known truth. H0 is not a new arm and need not be evaluated here.

Freeze lambda_C at the previous high value 0.24202997768915402; verify it against previous calibration.json and B_H1_high/result.json. Do not recalibrate from H1 or new gradients. lambda_TV=0.1, same original Direct TV on ImageNet-normalized pixels. The only changed objective coefficient is lambda_F, 1 in the stored control and 0 in the new arm:

    L_new(x) = 0 * L_FIA(x, target_z) + 0.1 * TV(normalize(x))
               + 0.24202997768915402 * L_C(DeltaTheta(x,H1)).

FIA target cosine can still be measured as a public diagnostic, but its derivative must not contribute to the optimizer step. No feature-norm penalty, clipping of feature norm, scale anchor, loss reweighting, midpoint attraction, changing tensor means or extra regularizer. TV remains active: this is FIA removal, not a completely prior-free cell-only run.

Keep signed symmetric nearest-even CPU FP32 INT4, q[-7,7], one observed scale per FC tensor, existing inclusive representable cell endpoints/caps and zero convention. Cell loss is the exact existing equal-six-tensor mean squared normalized outside-cell residual, chunks65536. All six unprojected actual image-generated FC updates are included. Scale equality remains a separate diagnostic.

VGG16 Conv frozen/eval; all six FC parameters trained during each local replay; public theta0 reset each replay; B16/T10, identical repeated full-batch order, SGD lr0.001, momentum=weight_decay=0, dropout inactive, mean hard-label cross entropy, raw after10-minus-start delta. Keep previous configure(), seed42, FP32, deterministic/TF32 settings and ImageNet normalization exactly once. No precision/backend fallback.

New arm uses identical fresh Adam on pixels, lr0.001 cosine to0.0001 over100 steps, betas(.9,.999), eps1e-8, weight_decay0, clamp [0,1] after each step. Start from original naive final pixels; no copied Adam moments and no resume from any screen endpoint. Run sequentially once; no best-iterate selection, restart after a completed real arm, learning-rate tuning, extra label vector, seed or longer continuation.

## Implementation and validation

Add isolated modules such as qacpa/signed_e2e_nofia_runner.py and signed_e2e_nofia_evaluation.py and focused tests. Reuse previous signed_e2e_cells.py, replay, codec and runner metric/normalization/schedule/helpers where appropriate without modifying them. Do not monkeypatch global flags or falsify the previous evaluator's task/arm metadata to pass its validator. Add a new explicit provenance validator for this task's stored-control/new-candidate comparison.

Protect all existing research source and prior outputs, including the completed signed_e2e_* modules and tests. Ordinary fixes to NEW task files and necessary focused rechecks are authorized. Hash previous inputs, control candidate/state/results/registration, shared helpers and new source before/after. The old control's candidate SHA256 in the prior summary is b352c3bde7db87ff00ecbbbd4715c5c865ada662c4dcf88b1e116f5369ec9d5a; verify from disk. A provenance/source change is a reason to stop and report, not to silently rerun the control or choose another artifact.

Focused validation must establish meaningful semantics: lambda_F=0 contributes no FIA gradient while actual cell and TV gradients remain; lambda_F=1 is consistent with the previous objective on a synthetic fixture; instrumentation cannot affect the optimizer gradient/state; H1/order/weights/initialization are immutable; stored-control and new-arm provenance plus the private-load phase boundary reject mismatches. Run relevant existing signed-cell/replay tests, but do not add implementation-mirroring tests merely for coverage or rerun unrelated suites without cause.

Before the new real arm, register the entire plan and hashes, all fixed settings and control reuse. Use the existing compare_replay_paths at x0/H1 to check exact canonical-versus-adjoint terminal deltas for all six tensors and features with/without grad mode. Stop on any discrepancy rather than changing numerical rules. Compare initial H1 cell loss/features to prior recorded calibration (same numerical computation path: 22.991680145263672 and feature norm21.787342071533203); record actual equality and discrepancies. Do not treat the separately accumulated CPU endpoint mean22.991682647381136 as the same numerical accumulator. A reproduction mismatch must be explained and resolved before claiming a one-factor comparison.

Record finite actual cell image gradient and resource usage. Use the existing chain through frozen Conv: numerical dL/dZ returned by fc_recomputed_gradients must be explicitly propagated through image->Z. Detach only where the existing adjoint contract requires it, not across the image path.

## Diagnostics, resource limits and comparisons

Per-step logging must distinguish pre-step losses from the post-step optimizer count. Keep existing fields: FIA diagnostic, TV, cell loss, base/total loss, actual LR, component/total image gradient norms, pixel step norm, clamp fraction and elapsed time. Separately report FIA, weighted TV, weighted cell gradient norms/direction where finite. FIA diagnostic gradients are read-only: the actual optimizer receives only 0.1*g_TV+lambda_C*g_C. If logging extra derivatives, verify on synthetic inputs that it leaves the update unchanged and release graphs/tensors appropriately. A zero norm yields undefined cosine with a reason, never NaN masquerading as zero.

After the new final image is frozen, perform canonical H1 replay for x0, stored B_H1_high and new C_H1_high_noFIA (three endpoint profiles). Authenticate stored-control metrics against the old profile; report any discrepancy. Preserve all six tensor profiles: cell loss, exact fixed-s code mismatch count/fraction, bounded-cell violation count/fraction, mean/max violations in s units and candidate scale; feature norm, target cosine, TV, pixel range and displacement from x0. Optionally obtain q==0 versus q!=0 mismatch partitions within these same endpoint passes using bounded chunks; do not add optimization or oracle selection. No repeated intermediate full endpoint sweeps are needed.

Exact feasibility requires zero cell violations AND zero fixed-s code mismatches over all six original, unprojected actual replay updates. A projected matrix is not evidence for this. Report scale reproduction separately; no relaxation of feasibility or invented image-success thresholds.

The single new arm is capped at100 optimizer steps. In addition to the declared preflight and three endpoint passes, no real optimization arm is authorized. Keep the previous resource ceiling of45 minutes for the real phase (preflight, new arm, replay diagnostics and offline evaluation combined), and12GiB CUDA peak allocated memory. Record times and memory even though the new run should be smaller than the previous five-arm screen. On OOM, nonfinite, numerical-path/provenance discrepancy, or deadline, preserve completed artifacts and report PARTIAL/NEEDS_REVIEW/FAILED accurately. No automatic changed-condition rerun or hidden expansion.

Only after final candidates/hashes are frozen may the new evaluator access the existing private original images, for image evaluation only. No private labels, original feature targets, true updates or oracle tuning in attack. Bundled labels/z may materialize when loading evaluation_truth.pt; disclose labels_consumed=false, and consume x only. Reuse installed metric weights, not a newly downloaded metric network.

Evaluate all16 images for x0, stored B_H1_high and new C_H1_high_noFIA with one Hungarian pixel-MSE correspondence per candidate set reused for PSNR/SSIM/VGG LPIPS. Save per-image and mean metrics and a figure showing original/x0/control/new. Stored control metrics should reproduce the prior evaluation; document differences if any and do not tune against them. Private evaluation is never used to choose labels, lambda, a checkpoint, stopping time, or a next arm.

Prespecified interpretation: compare lambda_F=1 vs0 at fixed H1 and numeric lambda_C. Report exact feasibility first and the entire public violation profile second. A drop in L_C alone is not proof that code/cell constraints were satisfied. Nonworse on loss, mismatch fraction, violation fraction and maximum s-unit violation with a strict improvement in at least one is descriptive public-fit domination; conflicting metrics are MIXED. Report all three image metrics, and classify conflicting directions as MIXED. Findings are descriptive for this batch/seed and fixed labels; no statistical significance/generalization or class accuracy claims. If removing FIA improves fitting but not image quality, conclude only that the target term impeded public fitting here. If substantial violation remains, fixed-label uncertainty and other optimization effects remain unresolved. No automatic next experiment.

## Artifacts and one completion handoff

New public root: quantization_aware_cpa/results/signed_e2e_h1_nofia_B16_T10_20260916/
New private root: quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916/
Task receipts/report/logs: .ai/execution-results/CPA_SIGNED_E2E_H1_NOFIA_20260916/

Refuse unexpected preexisting outputs. Retain registration, input/source integrity manifests, original and reused-control hashes, tests/logs, invocation/exit records, full logs, candidates/states, endpoint profiles, comparison.json/results.json and report.md. Mark COMPLETED only after authorized implementation, tests, the bounded real run, evaluation and reporting finish; scientific feasibility can still be false. Record the archived control separately from the one new executed arm.

Follow WORK_MODE: do not poll with model calls. Launch any background wrapper hidden and persistent, register one non-model process-exit callback bound to the verified existing Executor, then end the turn. On completion finish only the authorized report/evaluation, ensure the linked report and receipts exist BEFORE sending the notice, deliver ONE final success/failure/partial notice to the verified existing Planner, and end. Do not create new AI sessions, relay turns, monitor agents or additional experiments. Planner ends after one verified queue receipt.
