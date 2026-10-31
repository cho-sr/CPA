# INT8: screen all eight randomized members at FIA5000, continue public top three to FIA25000

TASK_ID: CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917
PLAN_VERSION: 1
STATUS: RUNNING
TASK_KIND: USER_STEERED_EIGHT_MEMBER_FIA_SCREEN_THEN_TOP_THREE_CONTINUATION
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_EIGHT_ICA_FIA5000_TOP3_TO25000_20260917
PARENT_TASK_ID: CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917
REVIEW_REFERENCE: USER_REPLACED_FEATURE_AVERAGING_WITH_PUBLIC_FIA_SCREENING
ACTIVATION_POLICY: EXECUTOR_OWNED_TRANSITION_PRESERVE_RUNNING_PARENT_HANDOFF
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_int8_randomized8_screen3_fia_B16_T10_20260917
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_int8_randomized8_screen3_fia_B16_T10_20260917
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917/report.md

## User steering and precedence

User first said "그럼 그냥 8개 다 하는게 낫겠지", then specified the exact replacement: "ICA는 8개 다하고 FIA에서 5천iter에서 top3만 25k 까지". This supersedes the earlier feature-average/one-ensemble-FIA plan. Implement the eight complete ICA separations, screen each individual target for5000 FIA steps, freeze the PUBLIC top3, and continue only those3 to a TOTAL of25000 steps each (20000 additional, not25000 additional). Keep the already authorized center INT8 baseline to25000 for comparison. Do not invert an averaged feature target or average final images in the new experiment.

The user explicitly selected the ranking rule in response to Planner's question: "특징 일치 손실 + 0.1×이미지 TV 손실이 낮은 순 (권장)". Therefore rank by the existing full FIA objective (feature cosine distance +.1 normalized TV), fixed before execution. Ground truth, LPIPS/SSIM/MSE/PSNR or human inspection of private references never determines the top3. One member/run means a whole16-image batch, not a single image. Select three of the eight member runs, not three images per slot or three of the baseline+members.

The completed portion of the prior request authorizes the same INT8 observation preparation and all necessary isolated implementation/validation. This new steering authorizes the additional individual FIA prefixes/continuations and the corresponding budget. No additional permission is required for that concrete scope. Keep all other fixed scientific conditions and all historical INT4 artifacts unchanged.

Read the immutable parent plan .ai/execution-handoffs/CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917_v1.md in full, SHA256318a67e7e62a021ea73865cc2917e909c78810b1505509d131563f9bb0388092. Its sender/public/private boundary, input/checkpoint/core pins, INT8 codec and exact bounded-cell sampler/one-public-anchor policy, M8 seeds, GD4000 settings, protected-source requirements and no-oracle evaluation rules are inherited. THIS document takes precedence over its aggregation, number of FIA runs, stage orchestration, output roots and90-minute budget. Do not execute both conflicting plans.

## Ownership transition and known parent failure

Existing designated Executor remains Luna/xhigh thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236,start1789518279030ms. Planner remains thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95,PID24972,start1789518271364ms. No new AI session/model or peer rebinding.

At Planner's one user-requested status check, .ai/EXECUTION_HANDOFF.md still marked the parent RUNNING/claimed by that Executor. Parent execution_summary.json reported PARTIAL: registration succeeded, sender failed before collect() ran with TypeError: collect() got an unexpected keyword argument 'command'. Parent wrapper19124 exited1 after about3.34 seconds; its exit callback was already queued to the same Executor. Public output held only registration.json; no ICA/FIA had started at that observation. Preserve this evidence and correct the parent status truthfully; no scientific outcome exists from that failed invocation.

Planner deliberately leaves the claimed parent handoff unchanged. As the single owner, Executor must inspect current state once, archive the parent handoff/results and any source revision needed for provenance, and atomically adopt THIS READY file into .ai/EXECUTION_HANDOFF.md when the previous wrapper is no longer active. This is a new user steering request, not permission to duplicate successful stages. Do not overwrite a running parent's plan from another process. If an ordinary parent repair has started since Planner's check, safely serialize the transition: preserve completed compatible observation/ICA/center-FIA artifacts and reuse them after authentication; do not run a second competing wrapper. Do not launch the superseded aggregate-FIA stage. If an incompatible stage has already started, retain its evidence and transition at a safe checkpoint without relabeling it as a member-screen stage; report the actual cost/state.

The sender CLI parsing failure is an ordinary implementation defect. Fix the isolated integration and validate that the actual parser->main->collect call does not pass the subcommand field; repair analogous entrypoint issues if found. Use a new registration/output tree rather than editing a registered code hash in place. No completed research optimization needs rerunning based on the current evidence. Do not stop solely because the previous instruction prohibited automatic scientific reruns: this user explicitly replaces the plan, and repairs necessary to execute it are authorized.

## Common scientific settings

Same archived B16/T10 single-client six-FC raw delta, VGG16 checkpoint and frozen Conv, no new local training, no changed labels/batch/SGD/data/normalization. Sender only converts the authenticated archived raw update to deterministic per-tensor signed symmetric INT8 q[-127,127], s=FP32(absmax/127), nearest-even. Sender alone may consume true updates. Attack receives public q/s only; all source checks, noise, ranking and continuation are public-only. Private x is consumed only in final separate evaluation, with labels/true z/true updates unused there.

Use exactly one center separation and eight random-member separations (or authenticated completed compatible outputs). Noise seeds1234+1000003*m for m0..7, cell-bounded FP32 sampler and one public extremal scale anchor inherited unchanged. Each matrix has its own whitening shared by its FastICA and GD stages. Unchanged signed_combination_ica.separate_observation, CPU FP32/threads1/seed42, FastICA max200/tol1e-6, GD4000 plain SGD lr.001/momentum0/decay0 and the same four prior weights. No early stopping/tuning/extra M/seeds/private source feedback. Preserve native signed source S_m[16,25088] and abs target and all logs/hashes.

For consistent source-slot initialization across member FIA runs, use the parent public center-reference absolute-cosine Hungarian permutation and sign alignment, fixed from signed recovered sources only. Do NOT average features. Unlike the averaged-feature method, no row-norm rescaling is needed here: preserve each member row's original magnitude and apply permutation/sign only. Store native and aligned tensors, mapping, signs, matched cosines and exact hashes. Inversion target is exactly abs(aligned_S_m) in reference order, finite FP32[16,25088]. Center target remains abs(S_center). No member filtering on cosine and no private alignment. Record this updated policy before screening.

## Staged FIA: same25000-step schedule from the start

All nine branches use the same original Direct image optimizer/loss/model: seed42, CUDA FP32, Adam initial lr.1, cosine scheduler horizon T_max=25000 from construction, feature weight1, normalized-image TV weight.1, identical random image initialization policy. Reset seed/runtime before constructing EACH branch and hash the ACTUAL initialized image parameter. Do not substitute a separately generated dummy initialization hash. Preserve adapter runtime settings and model construction/seed reset order from upstream_naive.run_upstream_fia.

IMPORTANT: setting the original FIA n_iter to5000 would exhaust the cosine schedule and make subsequent continuation a different experiment. Always construct Direct with n_iter_fi=25000 and let a separate outer loop stop at5000. Reuse UNMODIFIED official Direct.step/get_attack_state/set_attack_state and the original VGG16FeatureAdapter/weights. Add an isolated staged driver; do not modify or monkeypatch official Direct, upstream_naive or its protected _optimize/_experiment functions to bypass fingerprints.

Screen member00 through member07 sequentially for exactly5000 updates each. Save an immutable5000-step checkpoint for each containing raw optimizer image parameter (not exported clamped pixels), Adam moments/step state, scheduler state/current lr/T_max, completed_iterations=5000, RNG states where applicable, target/config/source hashes, runtime and logs. Save each corresponding boxed pixel candidate separately. Preserve raw state through endpoint reporting: scoring/export must not change the checkpoint image, optimizer/scheduler or RNG state. Keep all eight5000 checkpoints unchanged after selection; continuations write new child outputs with parent checkpoint hash.

The center control runs the same schedule to25000, retaining its5000 checkpoint/candidate for equal-budget comparison; it is not in the top3 competition. A compatible existing center25000 trajectory may be reused only with exact target/config/initialization/runtime/source binding and a genuine5000 checkpoint. Otherwise perform the one registered center run. Do not invent a5000 state from its final image or restart a completed center to get a missing checkpoint; disclose any permitted reuse limitation.

## Frozen public ranking at5000

Compute the score once on each member's state AFTER exactly5000 updates. Original logged loss from its final step is pre-update; do not mislabel that as the post5000 endpoint score. In no-grad/eval mode, take a CLONED image parameter, apply the SAME normalized-space box used by Direct.step, and compute:

score_m = mean_i[1 - cosine(f(x_m,i), target_m,i)] + 0.1*TV(x_m).

TV must be the unchanged Direct.total_variation on normalized image coordinates. Use the same feature model/dtype/reduction/cosine epsilon/runtime for every score. Preserve the original raw checkpoint state without in-place clamping. Store scalar and per-image feature distances, TV, total, checkpoint/boxed-score-image/target/source hashes and exact completed step. This is a public optimization-fit score; scores for different targets do NOT establish actual-image fidelity. Record that proxy limitation explicitly.

Require all eight screen stages and all scores finite/valid. Sort by (score ascending, member_index ascending), choose exactly the first3. Save selection.json with the complete eight-way ranking, deterministic tie rule, selected IDs and all evidence hashes BEFORE any selected continuation or private loading. No threshold adjustment, manual reranking, best-looking selection, private metric scoring, additional draws or replacing a failed selected branch with rank4. A failure produces truthful partial results and stops dependent selection/continuation as appropriate.

Resume only those3 from their authenticated5000 checkpoints for20000 more updates, ending at TOTAL25000. Restore image parameter/Adam/scheduler state exactly; validate completed step, current LR/T_max, optimizer parameter binding and hashes before the first resumed step. Do not restart initialization, Adam, cosine schedule, or use exported/clamped pixels as the resume state. Rank order is frozen; do not switch branches during continuation. Five unselected members remain at5000. Save all three25000 final candidates/logs/checkpoints and parent links.

## Required focused validation before the real pipeline

Inherit relevant INT8 codec/sampler/shape/integrity/private-boundary tests from the parent. Validate the actual corrected sender CLI dispatch with a mocked collect function, not only CLI --help or pure collector tests. Check new source member loaders accept precisely bound member records rather than treating all non-center sources as an aggregate. Exact abs/alignment/target binding must hold for all8 branches.

Add a meaningful split-versus-uninterrupted test using the SAME original Direct class and a small synthetic feature model/runtime: run N steps continuously versus K then checkpoint/save/load then N-K with the same TOTAL scheduler horizon. Require matching parameter, optimizer moments/step, scheduler state/lr and subsequent losses within a predeclared appropriate deterministic test tolerance (exact on a deterministic CPU fixture where feasible). Exercise actual serialization, stage APIs, and mutated target/step/scheduler/source rejection. No real extra full25000 validation run or full-size pilot.

Validate ranking using fake finite scores, ties, fewer/more than8 records, wrong checkpoint/source hashes and nonfinite cases. Assert exactly3 continuations, no continuation before all8 scores/selection are frozen, five members stay at5000, total counters25000 not30000, no parameter/state mutation by poststep score/export, and private evaluation cannot load before all public outputs are frozen. Capture REAL constructor initialization hashes for common-start validation. Test writer/loader/resume integration and stage budget accounting rather than relying only on shape mocks.

Snapshot all reused/new execution sources and the parent failed-source revision, run only relevant focused tests/py_compile/CLI help, and register final exact schemas/config/seeds/selection/scheduler/stop/resume policies and hashes before real computation. Protected historical INT4/ICA/FIA/model/metric sources remain unchanged. Ordinary isolated repairs are authorized; successful observations/separations/optimizer prefixes are preserved and authenticated for reuse rather than recalculated.

## Offline evaluation and presentation

Freeze center and all8 member5000 outputs, selected IDs, three25000 outputs and integrity before PUBLIC_GATE_PASSED. Only then run a separate evaluator with configure_runtime(42,threads=1), the unchanged archived private x-only truth and cached metric weights. Disclose bundled deserialization versus actual fields consumed. No private feedback enters the now-completed public attack.

Evaluate the registered frozen sets: center5000 and all8 member5000 (9 sets), plus center25000 and selected3 member25000 (4 sets), total13 sets if all saved as prescribed. Use one pixel-MSE Hungarian assignment per candidate reused for all MSE/PSNR/SSIM/VGG LPIPS and the aligned figures. Do not average images, privately choose the winner or collapse the three selected runs into an oracle-best result. Primary final contrast is the public5000 rank1 member at25000 versus center25000; ranks2/3 are separately reported secondary contrasts. Also report each selected member's5000->25000 change and all8 screen outcomes descriptively. Private outcomes must not change the frozen public ranking or trigger a rerun.

Produce a5000-step contact sheet (truth, center, all8 members) and a25000-step sheet (truth, center, public ranks1/2/3) covering all16 images. Public files contain no truth; private metrics/figures stay under PRIVATE_OUTPUT_ROOT. Exact actual-image cell feasibility is NOT_EVALUATED because no FC replay/cell refinement is included. Good public FIA fit does not establish correct source separation or image recovery, and public-top3 selection is not oracle selection.

## Budget and scheduling

FIA update budget: 8*5000 + 3*(25000-5000) =100000 member updates, plus25000 center updates =125000 total. At previous measured about1994 seconds/25000 updates, FIA alone is roughly166 minutes; separation/checkpointing/scoring/evaluation add time. Planner estimates about3 hours, not a measured promise. Use a new aggregate cap12600 seconds (210 minutes/3.5 hours) for required real sender/separation/FIA/scoring/evaluation stages, including any compatible real work reused after user steering; development/focused tests/idle excluded. CUDA allocated cap remains12GiB=12884901888 bytes. No silent device/precision/iteration/ranking/budget changes or extra arms on failure.

Register the entire serial stage order and complete persistent wrapper BEFORE launching: sender if needed -> center+8 separations -> public alignment -> eight5000 screens -> frozen selection -> selected3 continuations -> center baseline trajectory (may precede screening if registered consistently) -> public integrity gate -> one offline evaluation. Separate sender, public attack stages and private evaluator process roles. Charge actual stage durations separately and cumulative elapsed time correctly; the parent's old wrapper mislabeled cumulative elapsed as stage wall time, which must not be repeated. Preserve no-overwrite stage paths and immutable checkpoints.

Launch hidden/persistent with exact PID/start verification and one non-model OS exit callback to the SAME Executor; check one startup receipt then END the turn. No polling/new agents. Use BOM-free UTF8 and do not alter shared peer bindings. Completion notification may finish authorized evaluation/reporting repairs from saved results, then send one final result to existing Planner and end. No unselected-member continuation, repeated screen, new sample count or further experiment follows automatically.

Final report/execution_summary must include parent failure and ownership transition, scope/selection rule, exact phase-specific data consumption, source/input/output integrity, sender/code/scale checks, all9 ICA results, actual common FIA initialization, eight public5000 scores and rank3 selection, checkpoint/resume evidence, exact iteration totals, all registered private metrics/figures, scientific limitations and actual timing/resources/exits. Mark partial/needs-review honestly when required stages fail; do not label a missing image-cell test false.
