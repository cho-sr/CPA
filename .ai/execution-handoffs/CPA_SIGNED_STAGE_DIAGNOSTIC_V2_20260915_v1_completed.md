# Implement and execute the approved saved-result stage diagnostic

TASK_ID: CPA_SIGNED_STAGE_DIAGNOSTIC_V2_20260915
PLAN_VERSION: 1
STATUS: COMPLETED
TASK_KIND: DIAGNOSTIC_IMPLEMENTATION_TESTS_AND_ONE_SAVED_RESULT_RUN
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: 01a09e4d-51f3-7121-84e2-03a7a9ac3039
COMPLETED_BY: 01a09e4d-51f3-7121-84e2-03a7a9ac3039
RESULT_STATUS: COMPLETED
FULL_EXPERIMENT_RESULT: ONE_SAVED_RESULT_DIAGNOSTIC_COMPLETED_32_CANDIDATES
PLANNER_FINAL_DELIVERY: FINAL_RESULT_HANDOFF_CONFIRMED
PLANNER_FINAL_RECEIPT: Queued message 01a0a467-fc0c-71f1-9818-b08d51d6d6e9 to thread 01a09ac2-0e8d-7f90-94a8-c45be99f4343
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_HAND_TO_LUNA_AND_EXECUTE_DIAGNOSTIC_20260915
DIAGNOSTIC_EXECUTION_AUTHORIZATION: IMPLEMENT_TEST_AND_RUN_V2_ON_EXISTING_32_IMAGES
RECONSTRUCTION_RERUN_AUTHORIZATION: NONE
PLANNER_SESSION_ROLE: CODEX_PLANNER
METHOD_PLAN: .ai/proposals/SIGNED_CPA_FIA_STAGE_DIAGNOSTIC_20260915.md
METHOD_PLAN_VERSION: 2
METHOD_PLAN_SHA256: d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d
CONSENSUS_REFERENCE: CLAUDE_TURN54_ACCEPT_WITH_CHANGES_AND_CODEX_V2_QUALIFICATIONS
PARENT_TASK: CPA_SIGNED_CELL_EVALUATION_REPAIR_20260915
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_STAGE_DIAGNOSTIC_V2_20260915/report.md
MODEL_DECISION: Same existing user-designated Luna/xhigh Executor

## 1. Actual latest authorization and task identity

[FACT/user] After the completed comparison, the user requested a diagnostic experiment plan and Claude review. Claude Turn54 accepted with changes. Codex incorporated the four changes and mathematical/interpretation qualifications in method plan v2. The user has now explicitly instructed:

> 루나한테 넘겨서 실험 실행

This authorizes implementation, meaningful tests, and the single saved-result diagnostic defined in v2. The proposal's PROPOSAL_ONLY / execution-not-requested wording describes its earlier planning state; this newer explicit instruction authorizes the bounded task NOW. Do not ask again for this scope or substitute another planning-only response. The method plan's scientific definitions, tests, resource limits and exclusions remain binding. Claude reviewed v1 and proposed A-D; do not misstate that Claude reviewed the final v2 wording itself.

Recipient: existing Luna/xhigh thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039, PID18272, creation /Date(1789362262182)/, workspace C:/Users/조상제/Desktop/CPA. Verify own identity/current runtime, handoff task/version/pre-claim SHA and the method plan SHA before claiming READY as RUNNING. Follow required project docs and role split. Historical Astra/high bridge wording does not override the user's Luna selection; no model/binding change, replacement CLI or new AI worker. The preceding evaluation-repair task is COMPLETED; archive/history stays preserved.

## 2. Deliver the complete approved implementation

Read the full v2 method plan and completed evaluation repair report before implementation. Implement its section4 measurements, A-D revisions, section5 outputs and section6 synthetic tests, in a separate diagnostic path:

- NEW quantization_aware_cpa/qacpa/signed_stage_diagnostics.py
- NEW quantization_aware_cpa/tests/test_signed_stage_diagnostics.py (additional focused test file only if useful)
- Optional NEW quantization_aware_cpa/docs/signed_stage_diagnostics.md for actual CLI/schema documentation
- Execution-only wrapper, registered plan/source/input manifest, logs, measurements and report under this task's result directory

Keep existing original CPA/FIA/eig/whitening/optimizer/loss/adapters/codec, projection core, scientific runner, completed evaluator and earlier tests intact. The new evaluator may call existing compatible provenance/read utilities. Do not weaken validators, alter recorded metadata to pass them, or copy a permissive validator. An evaluator that requires a new output path must receive the real new diagnostic path; do not pass the old evaluation output and bypass its existing-output check.

Minimal new diagnostic implementation/CLI/plumbing fixes and necessary regression checks within this same approved scope are authorized. Resolve routine packaging/import issues without re-requesting execution permission. A true provenance mismatch, missing binding evidence, or a fix requiring a scientific-condition change must be reported honestly instead of bypassed.

Do not stop after tests: after checks pass, execute the real saved-result diagnostic once and produce the requested numerical report. Tests must use tiny synthetic data and exercise actual normalization, mapping joins, geometric bounds, sign mixing, zero-norm handling and private-load boundaries; do not mock away the function under test. Include a mocked end-to-end CLI/path/schema test before real evaluation. Run the new focused suite and the existing relevant signed-cell regression suite once; broaden testing only for a new relevant failure/edit. Use the DLG Python and correct package paths as documented by the prior successful test invocation.

## 3. Concrete CLI contract to implement, then execute

The module is new; this is the required interface to implement and test before calling it. From repository cwd:

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_stage_diagnostics --naive quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive --cell-rank quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5 --comparison quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5/comparison.json --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --private-observation quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation --checkpoint "C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth" --method-plan .ai/proposals/SIGNED_CPA_FIA_STAGE_DIAGNOSTIC_20260915.md --expected-plan-sha256 d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d --output quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2 --device cuda --eval-microbatch 1 --max-evaluation-seconds 600 --max-cuda-allocated-gib 6

Focused test command after implementation (set process-local PYTHONPATH correctly without modifying shared settings):

    C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_signed_stage_diagnostics.py quantization_aware_cpa/tests/test_signed_cell_rank.py quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py quantization_aware_cpa/tests/test_signed_cell_rank_validation.py -q --basetemp .tmp_pytest_signed_stage_diagnostic_v2_20260915

The numeric geometry tolerance is fixed by the method plan at1e-10; record it in configuration. Runtime flags must not permit silent deviation from scientific inputs/encoder, the frozen assignment maps, or the count/shape of32 candidates. No automatic fallback to a different inference definition or rerun of old attacks. If the exact output or test temp directory already exists, inspect ownership and preserve it; never delete existing directories or duplicate an already completed diagnostic. Routine test-temp naming can use a unique task-local subdirectory and be recorded without changing the research output path.

## 4. Measurements and interpretation that must remain explicit

All32 final saved candidate images (16 naive,16 AP5), each paired with its actual supplied target_z in the same source order. Authenticate source/candidate hashes and generating convention before inferring anything from range. Verify checkpoint file SHA256 and actual loaded features state against its features parameters; record normalization once with the exact ImageNet mean/std and features->avgpool->flatten definition. No extra clamp/resize/augmentation.

Compute h=f(normalized saved pixels), t=existing abs(S_hat), and reuse saved evaluation-only oracle z. C_fit=cos(h,t) uses no matching. Primary pi_I is the already frozen image-MSE assignment; secondary pi_E is the already frozen embedding-cosine assignment. Under EACH named map use one coherent candidate/truth triple for target/oracle and image-feature/oracle measures. Preserve both maps, never rematch h, never select the more favorable view, never pair ICA source numbers across arms as if shared. Reuse LPIPS/PSNR/SSIM/MSE rows from comparison.json with explicit candidate/truth keys; no metric recomputation or ground-truth ranking.

CHANGE A: per-candidate bounds lower/upper=a*b +/- sqrt((1-a*a)*(1-b*b)), observed c and c-b, exact v2 finite/clamp/null/degenerate-interval handling. Bounds are mathematical consistency, not independent evidence or a principal-angle/optimization gate. c-b describes change in cosine to the same oracle; it is not a causal effect.

CHANGE B: validate bijective maps; report agreement/disagreement counts/fraction and all IDs. The disagreement fraction is NOT identity accuracy or success rate.

CHANGE C: actual pixel producer/schema and checkpoint/loaded-feature verification before comparing features. Use original final saved pixels; do not replace them with the historical pre-optimizer log state. Record any log/final measurement difference descriptively.

CHANGE D: source sign energies, counts and min(E_pos,E_neg)/(E_pos+E_neg), invariant under global sign reversal; retain orientation-dependent negative energy fraction as raw context. Best-global-sign abs cosine and relativeL2 follow v2 equations. All-negative rows can have zero sign mixing; large negative mass alone is not adapter damage. No sign flip or alternate adapter is fed to FIA, and this diagnostic does not establish whether abs helps/hurts true identity.

Unknown zero-norm cosines use JSON null with reasons and valid/undefined counts; all32 rows retained. Correlations are descriptive within arm with n<=16 and tied/constant handling, no significance/generalization or new thresholds. Retain H_TARGET/H_FIA/H_MIXED limitations and any competing interpretation. Do not force a unique CPA/FIA cause merely from aggregate means or from the diagnostics' algebra.

## 5. Information and artifact boundaries

Preserve original signed deterministic INT4 per-parameter FP32 scalar scale/q[-7,7]/rounding/zero/bounded-cell semantics; VGG16 Conv-frozen FC-trained single-client B16/T10 seed42; same SGD/rules/raw delta/classifier.0.weight; original CPA/FIA25k and AP5 uncentered rank16 gram_eigh. Scale7.14% drift is historical metric-only evidence. No new scale constraint, losses, whitening changes, MP/noise assumption, seeds/arms, local training/replay, input generation or AP5/CPA/FIA execution.

Load evaluation_truth.pt only after public/finalized-arm/comparison/checkpoint binding validation. Consume only oracle z; no x or labels need to be accessed. The serialized file bundles x/labels/z, so disclose possible materialization by torch.load accurately; never claim physical non-loading of labels while loading the bundled file. labels_consumed=false. Do not deserialize raw_updates.pt for this diagnostic; already computed true-G errors are context. Hashing files for provenance is distinct from tensor deserialization.

Pre-register task/version, v2 plan/hash, exact command/configuration, source/input/checkpoint/comparison hashes and runtime policy before research tensor evaluation. Preserve all original artifacts, source/fingerprint evidence and previous output/status history. Compare protected originals and all reused input/source/candidate/comparison bytes before/after. signed_cell_rank.py stays7ae6deb49738914f4cd1f9328397400d89835ede4cd6a462b2ea260a77ae4acb; completed evaluator staysc163aaa54bd8d5ebfc1528f49cafe2a830de72a9d7fe38583391b7ddaf4a65ef. Do not modify the referenced v2 plan after hash verification.

## 6. Finite cost, output and lifecycle

Real diagnostic budget: one paired analysis, exactly32 final-image forward evaluations, evaluation microbatch1, frozen/eval FP32 features, TF32 disabled, no autograd/optimization. The original client batch remains16. A-D summaries use those same features and saved vectors. Max600seconds diagnostic wall time and6GiB CUDA allocated memory; enforce within the diagnostic process and record actual wall/peak allocation, reporting CPU tests separately. Use a fresh process so baseline CUDA allocation is measurable; report start/peak/delta and allocator metric definition. Ceilings are not estimated completion time. On OOM/nonfinite data/provenance violation/budget exhaustion stop safely, preserve outputs and report; do not change conditions or auto-restart. No additional diagnostic run or feature sweep is authorized.

Outputs from method plan: registered_plan.json, diagnostic.json, all32_per_candidate.csv, paired16_by_truth.csv, plot and report.md under the new stage_diagnostic_v2 directory. The task result folder also contains test/syntax/runtime logs, exit codes, integrity evidence, final report.md/results.json. Intermediate calculated features may be saved in this new diagnostic output for traceability and to avoid repeated inference, with explicit hashes and evaluation-only metadata. Write numerical outputs before plotting; a plot error must not erase valid numerical results.

Follow WORK_MODE.md: run required work to completion, but do not keep a model polling/sleeping. If background execution is needed, use a hidden execution-only wrapper with lifetime preserved, exact PID/start-time record, serialized exit guards and at most one model-free completion callback via .ai/completion_notify.py to this same Executor. Verify preparation once then end the model turn. Short commands may finish immediately; no unnecessary callback or waiting agent. The callback may finish authorized evaluation validation/reporting, without rerunning completed stages.

## 7. Final report and single notification

Final report must show actual implementation files/changes, meaningful tests and exit codes, actual diagnostic command/status and number of evaluated images, normalization/encoder/provenance checks, raw32 rows and paired16 rows, mean/median/range of final fit and both oracle views, mapping agreement/disagreement, geometric consistency/c-b summaries, sign mixing/abs geometry, undefined counts, descriptive associations, historical log comparison, runtime/VRAM/budget outcomes and preserved input/source hashes. Link machine-readable results/plot. Separate FACT/INFERENCE/PROPOSAL/UNKNOWN, with no ground-truth selection or unsupported causal claim.

Mark COMPLETED only after implementation, tests, real diagnostic and required verification/report finish. On failure state exact finished stages and remaining work; no fabricated measurements. Update active handoff claim/status/result paths; leave historical completed tasks and RELAY/STATE unchanged.

Send final completion/failure once to existing Planner thread01a09ac2-0e8d-7f90-94a8-c45be99f4343 PID12728 after verifying its pinned process identity. Use existing queue delivery, record receipt, no blind retry/rebind/new CLI/courier reply/Claude discussion. The existing user-authorized one-time final-result notice applies. Planner displays the result once and ends its turn without monitoring.
