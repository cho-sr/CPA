# Repair the saved-result diagnostic and specify the next latent-G extension

TASK_ID: CPA_SIGNED_STAGE_DIAGNOSTIC_REPAIR_20260915
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: DIAGNOSTIC_REPAIR_TESTS_ONE_SAVED_RESULT_EVALUATION_AND_DESIGN
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_GO_AHEAD_AFTER_DIAGNOSTIC_REPAIR_AND_LATENT_G_ROADMAP_20260915
RECONSTRUCTION_RERUN_AUTHORIZATION: NONE
LATENT_G_OPTIMIZER_RUN_AUTHORIZATION: NONE_IN_THIS_HANDOFF
PLANNER_SESSION_ROLE: CODEX_PLANNER
PARENT_TASK: CPA_SIGNED_STAGE_DIAGNOSTIC_V2_20260915
METHOD_PLAN: .ai/proposals/SIGNED_CPA_FIA_STAGE_DIAGNOSTIC_20260915.md
METHOD_PLAN_VERSION: 2
METHOD_PLAN_SHA256: d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_STAGE_DIAGNOSTIC_REPAIR_20260915/report.md
MODEL_DECISION: Same existing user-designated Luna/xhigh Executor

## 1. Authorization, identity and concrete scope

[FACT/user] After the user asked for the implementation roadmap, the Planner explained that the saved-result diagnostic needs repair, while the full CPA-objective latent-G optimizer is not implemented. The user then said "ㄱㄱ". The Planner stated the immediate action: repair the S_hat/target_z mix-up and mapping errors, evaluate the existing saved results again, and specify a latent-G optimization design that preserves original CPA code. This handoff implements that bounded next step. Do not ask again for the authorized repairs, tests and corrected evaluation. Older PROPOSAL_ONLY wording in the diagnostic method plan records its earlier planning status.

Recipient is the existing Luna/xhigh thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039, PID 18272, creation /Date(1789362262182)/, workspace C:/Users/조상제/Desktop/CPA. Verify identity, actual settings, READY/unclaimed task/version and delivered snapshot SHA before claiming RUNNING. Historical Astra/high wording does not override the user's Luna selection. Do not change models, bindings or global settings, start a new AI worker/CLI, or take Planner relay turns.

Deliver: diagnostic repairs, meaningful CPU synthetic/mock/negative tests, ONE corrected saved-result evaluation on the same 32 images, a numerical report, and a source-grounded latent-G design document. Complete the real evaluation after tests pass. Do not implement or launch a new latent-G optimizer in this handoff: its actual objective/gradient compatibility is a required design deliverable, not a settled scientific setting. Do not silently replace the user's latent-G objective with the already tested AP5 proxy.

## 2. Confirmed defects to repair

Read the full immutable diagnostic method plan v2 and relevant code. Preserve its scientific definitions. In quantization_aware_cpa/qacpa/signed_stage_diagnostics.py the current code:

1. Checks saved target_z == abs(S_hat), but discards target_z and uses signed S_hat for C_fit and C_target_I/E. Store targets separately per arm. Compute C_fit=cos(h,target_z), C_target_I/E=cos(target_z,z under the named frozen map). Signed S_hat remains exclusively the input to sign diagnostics. Do not globally overwrite signed sources, take abs of a final cosine, or repair aggregate means algebraically. Correct all dependent geometry, summaries, correlations and plots from actual vectors. With this VGG ReLU feature/target definition, valid nonzero h and target_z imply a nonnegative cosine; include an appropriate validation guard and explicit zero-norm behavior.
2. Reports equality of naive and cell source-index mappings as mapping agreement. Instead, within EACH arm compare that arm's frozen image-MSE map pi_I against its frozen embedding-cosine map pi_E. Report count, fraction, disagreement and complete IDs separately for naive and cell. Different arms have unrelated ICA source numbering. Do not call agreement an identity accuracy.
3. Pairs rows across arms by pi_I truth, then subtracts C_image_E whose truths can differ. Maintain separate I and E same-truth paired views. The existing paired16_by_truth.csv can be the clearly named I view; add paired16_by_embedding_truth.csv for E. Each row must expose candidate IDs and truth IDs proving that both operands use the same truth under the stated map. Image MSE/LPIPS/PSNR/SSIM belong only to the existing I map unless an already saved value supports another view; do not fabricate E-view image scores or recompute matching.

## 3. Restore the already requested v2 validation contract

Keep this repair confined to the diagnostic and its tests/docs. The previous helper-only tests failed to detect actual data-flow defects. Restore these method-plan requirements in the repaired path:

- Reuse signed_cell_rank_evaluation.validate_provenance for finalized arm/sender/public/checkpoint/runtime/initialization/source bindings before any private tensor deserialization. It hashes raw_updates.pt without deserializing it, which is allowed. Pass the real fresh diagnostic output directory to path validation before creating it. Do not pass a historical output path, weaken a guard, or rewrite metadata to pass.
- Additionally bind the supplied completed comparison.json, its provenance and arm final_result records and frozen candidate/truth mapping rows to those exact validated inputs. Account for documented mutable evaluation status/path fields explicitly; do not demand meaningless whole-object equality across different output paths. Wrong sender, observation, checkpoint, arm, artifact hash, duplicate/missing map row or finalized status must fail before loading oracle tensors. Keep the completed evaluator source unchanged.
- Verify actual artifact schema, shapes and FP32 dtype before use; do not silently cast an invalid dtype to satisfy the contract. Keep normalization exactly once, saved final pixels in [0,1], no additional resize/clamp/augmentation, frozen/eval FP32 features->avgpool->flatten, microbatch 1, TF32 disabled. Verify the checkpoint and loaded feature weights before interpreting feature/oracle comparisons.
- Preserve raw cosine/norm/finite evidence and zero-norm null reasons in the detailed rows. Apply the method plan's 1e-10 numerical tolerance policy; do not silently clamp an arbitrary invalid value into a valid cosine. Distinguish VALID/UNDEFINED/INVALID/VIOLATION in geometry totals instead of counting INVALID as valid with zero violations. No geometry or principal-angle attack gate.
- Enforce the existing real diagnostic 600-second budget globally across both arms and required diagnostic stages, not a fresh 600 seconds for each arm. Keep the 6 GiB CUDA allocated-memory limit and record actual total wall time and allocator definitions. No scientific change to recover from OOM or a true provenance failure.

Routine packaging, diagnostic CLI/schema fixes and report wiring in this bounded task are authorized. An unresolved evidence mismatch must be reported as UNKNOWN/failure, not silently bypassed. Preserve all prior code/results outside the allowed paths, including the incorrect historical diagnostic outputs as auditable history.

Allowed edits: quantization_aware_cpa/qacpa/signed_stage_diagnostics.py; quantization_aware_cpa/tests/test_signed_stage_diagnostics.py; an additional focused diagnostic test file if needed; quantization_aware_cpa/docs/signed_stage_diagnostics.md if present/useful; this task's execution-only wrappers, reports and new output directory. Do not alter original CPA/FIA/eig/whitening/loss/optimizer, codec, AP5 projection, scientific runner/adapters, completed comparison evaluator or existing non-diagnostic tests. Do not change the v2 method-plan bytes.

## 4. Meaningful validation before one real corrected evaluation

Add an integration test exercising actual run_diagnostic measurement and output wiring with small synthetic artifacts and a mocked encoder only where needed. Use mixed/negative signed S, different target_z vectors in each arm, and known nonnegative h; ensure it would fail under the previous signed-target bug and cannot accidentally reuse the last arm's target. Check global S sign reversal leaves fit/target metrics and invariant sign-mixing statistics unchanged while orientation-dependent sign statistics remain truthful.

Use deliberately different pi_I/pi_E permutations in both arms. Assert within-arm agreement and separate I/E same-truth pairing by explicit IDs, with reference dot products rather than assertions mirroring implementation. Include shuffled row order, duplicates/missing rows and a case where the former I-paired E delta is wrong. Ensure saved image scores are joined to their actual candidate/truth row.

Exercise real validation boundaries with synthetic metadata and spies on the private loader: malformed/missing bindings and comparison hashes/identities fail before oracle loading. Include dtype/schema/nonfinite/negative-ReLU-feature/zero-norm/numerical-boundary cases and a mocked global deadline crossing between arms. Do not mock away the validator or measurement under test. No real private artifacts in tests.

Run the focused diagnostic suite plus existing relevant signed-cell regressions, recording actual commands/counts/exit codes. Example from repo cwd, with the prior documented process-local PYTHONPATH as needed:

    C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_signed_stage_diagnostics.py quantization_aware_cpa/tests/test_signed_cell_rank.py quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py quantization_aware_cpa/tests/test_signed_cell_rank_validation.py -q --basetemp .tmp_pytest_signed_stage_diagnostic_repair_20260915

Additional diagnostic tests must be included if created. Preserve any existing temp directory; use a recorded unique task-local test temp path rather than deleting prior data. Repeat tests only for relevant edits/failures.

## 5. Exact saved inputs and corrected execution

Register source/plan/input hashes and exact configuration before the real diagnostic. All 32 saved final candidates are reused: 16 naive and 16 AP5. There is no new attack arm, seed, training, observation generation, local-SGD replay, AP5, CPA or FIA execution. The only real model inference is one pass over these 32 saved images; save computed h vectors and hashes in the new evaluation-only output for traceability. No automatic real-run retry.

Execute once, after tests pass, from repo cwd:

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_stage_diagnostics --naive quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive --cell-rank quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5 --comparison quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5/comparison.json --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --private-observation quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation --checkpoint "C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth" --method-plan .ai/proposals/SIGNED_CPA_FIA_STAGE_DIAGNOSTIC_20260915.md --expected-plan-sha256 d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d --output quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired --device cuda --eval-microbatch 1 --max-evaluation-seconds 600 --max-cuda-allocated-gib 6

The output must not preexist. Preserve stage_diagnostic_v2 and evaluation_ap5 unchanged. Label corrected results with a diagnostic implementation revision and superseded-metric provenance. Keep original image/embedding evaluation results unchanged; this repair does not invalidate their independently computed scores merely because the later diagnostic used the wrong target.

Consume only oracle z from evaluation_truth.pt after provenance validation. Disclose that torch.load of a bundled file can materialize x/labels, while labels_consumed=false and x is not used. Do not deserialize raw_updates.pt or recompute true-G errors. No GT use for selection, loss, tuning, stopping or new rankings.

Preserve signed deterministic INT4 q[-7,7], observed per-tensor FP32 scalar s=max_abs/7, existing tie/zero/cell rules; Conv frozen, FC trained, single client B16/T10 seed42, public starting checkpoint/rules, raw theta_after10-minus-theta_start classifier.0.weight update. No midpoint loss, scale loss/anchor, MP/dither/iid error assumption or condition change.

Compare original protected sources and all reused input/plan/checkpoint/candidate/source/comparison files before/after. In particular signed_cell_rank.py remains 7ae6deb49738914f4cd1f9328397400d89835ede4cd6a462b2ea260a77ae4acb and signed_cell_rank_evaluation.py remains c163aaa54bd8d5ebfc1528f49cafe2a830de72a9d7fe38583391b7ddaf4a65ef. Record hashes rather than rebinding preserved evidence.

## 6. Next latent-G design deliverable, with actual original CPA semantics

Write .ai/execution-results/CPA_SIGNED_STAGE_DIAGNOSTIC_REPAIR_20260915/latent_g_design.md. This is a concrete design for later implementation, not a claim of a new completed experiment. Preserve original source files and distinguish FACT/INFERENCE/PROPOSAL/UNKNOWN.

Inspect official_cpa/src/gradient_inversion.py and the actual saved CPA hyperparameters. The Planner confirmed the original code computes negentropy then overwrites loss_ne with a constant zero. Decorrelation is a function of normalized W_hat only. Do not activate negentropy, claim a direct G-gradient from decorrelation, invent an MI term, or edit the original code to resolve those facts. Identify active sign/sparsity/TV terms using actual configuration, their gradient paths to G versus W, and why full-rank ICA followed by its inverse is a no-op. Inactive/structurally zero gradient terms must be labeled, not represented as working optimization signals.

Specify how a separate optional extension could preserve baseline API/results, the exact CPA-compatible centering/mean restoration and original eig whitening semantics. Under fixed whitening K, reconstructed sources are W_norm K G; explain its low-rank nullspace and why W decorrelation alone does not guarantee independent/noncollapsed actual sources. Compare fixed and recomputed whitening with explicit derivative/stability/memory implications, without silently changing canonical CPA whitening or adding whitening to the loss.

Analyze bounded representable INT4 cells, nearest-even boundaries, q=+/-7 initialization at sq, zero-tensor cells, and tanh open-interval endpoint/inverse-atanh limitations. Do not present finite tanh parameters as exactly reaching an endpoint or silently change observed scale. No midpoint attraction or scale enforcement in the proposed default; scale remains a logged metric under current decisions.

Provide the smallest defensible prospective objective/parameter/initialization/optimizer/interface design, a gradient and degeneracy analysis, and a bounded future validation/ablation plan with expected memory/runtime costs or UNKNOWN estimates. If original active priors are insufficient to identify G, state that clearly and present alternatives as proposals requiring an explicit scientific choice; do not manufacture a selector or claim theoretical recovery from q,s. Maintain the user's requested latent-G direction and distinguish it from AP5 rank projection. Relate corrected diagnostic evidence to expected benefits/limits without GT-driven tuning, retrospective selection or causal overclaim.

## 7. Outputs, status and delivery

New diagnostic output: registered_plan.json, diagnostic.json, raw32 CSV, separate I/E paired16 CSVs, h vectors with evaluation-only provenance, plot and report. All summaries/plots must derive from corrected raw rows. Report fit/target/image cosines for both named views, mapping agreements by arm, zero/invalid/geometry counts, raw sign statistics, descriptive associations and all per-candidate values. Retain all 32 rows and report valid sample counts.

Task result folder: report.md, results.json, latent_g_design.md, exact tests and execution logs/exit codes, source/input integrity evidence, resource measurements, and an explanation of which old diagnostic quantities are invalid/superseded. The former negative C_fit means and cross-arm mapping agreements must not be reused as evidence that FIA failed. Separate valid historical image metrics from repaired diagnostic inference. Mixed evidence or remaining uncertainty is an acceptable result; no generalized attack success claim.

The previous execution record reported stage exit 0 while its completion callback recorded process exit 1. Preserve that history. In this run capture truthful stage and wrapper/process exits, propagating actual failure instead of calling all statuses zero. Do not expand into a generic completion-tool rewrite.

Mark COMPLETED only after authorized repairs, tests, the corrected saved-result evaluation, integrity checks and both numerical/design reports finish. If blocked/failing, identify exact completed stages and remaining work; preserve artifacts and do not auto-rerun or alter settings. Record claim/status and final report paths in the active handoff only; leave RELAY/STATE/history unchanged.

Follow WORK_MODE.md: do not keep a model waiting or polling. If execution is backgrounded, use one hidden execution-only wrapper with exact PID/start and, if needed, one model-free process-exit callback to this Executor. After preparation is verified once, end the model turn. No monitoring worker. On completion perform only the remaining authorized reporting and send one final result/failure notice to the existing Planner thread 01a09ac2-0e8d-7f90-94a8-c45be99f4343 PID12728 after pinned-identity verification. Record the receipt; no blind retry, new CLI or Claude loop. Planner displays the final result once and ends its turn.
