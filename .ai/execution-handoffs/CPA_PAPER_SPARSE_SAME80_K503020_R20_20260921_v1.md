# Paper ImageNet data: matched random filling at keep 50%, 30%, 20%

TASK_ID: CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: USER_AUTHORIZED_MATCHED_PAPER_SPARSIFICATION_SWEEP
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: max
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_PAPER_ASSET_DATA_KEEP50_30_20_RERUN_20260921
PLANNER_THREAD: 01a0bffa-40dd-7213-b387-7022055cec39
EXECUTOR_TARGET: USER_DESIGNATED_EXISTING_TERMINAL_LUNA
RESULT_REPORT_PATH: .ai/execution-results/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/report.md
SWEEP_OUTPUT_ROOT: quantization_aware_cpa/results/paper_sparse_same80_keep50_30_20_R20_20260921

## Authorization, priority and roles

User first requested "20% 실험 진행하자", then explicitly expanded the scope:
"데이터는 논문에 들어갈 paper asset 이 포함된 데이터셋으로 진행하고 50% 30% 20% 다시 하자".
The active scope is therefore random-fill reconstruction at retained fractions
50%, 30%, 20% on the exact existing paper data, using the matched 10% protocol.
This authorizes the necessary isolated implementation, checks, three conditions,
offline evaluation and saved-image exports. No further scientific approval is needed.
Percentages mean RETAINED coordinates, not discarded coordinates.

This session is CODEX_PLANNER / Astra xhigh. Research implementation and execution
belong to the user's existing terminal Luna. Do not start a replacement agent.
Local runtime on 2026-09-21 identifies a live terminal Luna PID9296, created
1789940454939ms, thread01a0c0c3-d687-7d92-824c-7499758967d5, cwd this workspace.
Latest settings evidence logs_2.sqlite row6507193 is Luna/max (previous xhigh was
row6507191). REQUIRED_REASONING_EFFORT records this actual setting, not a requested
global model switch. No saved first conversation was present at inspection.
Verify the live identity and settings again before claiming; never infer identity
from PID ordering. Historical Executor PID18908 and old Planner PID3348 exited.

The previous CIFAR-100 task was READY, unclaimed. It remains authorized and is
preserved byte-for-byte at
.ai/execution-handoffs/CPA_CIFAR100_ORIGINAL_CPA_PERTENSOR_BITS87654_20260920_before_sparse_20260921.md
SHA25623abd028dbc370242cd82f9b4e591dd7deb1d5ec9586e00e7c2614a199bd478a.
It is deferred in the task index, not cancelled or completed. This new explicit
request is current priority. Do not run CIFAR automatically from this handoff.
Check once for live conflicting research ownership; never kill unrelated work.

## Exact dataset and paper-image identity

Use the existing cached ImageNet validation sample at
quantization_aware_cpa/results/m6_uia_fp32_B16_fia/batch00..04/inputs.pt,
all original 80 identities, original five B16 batches, same ordering and tensors.
Use the saved updates/packets; no resampling, new data download, retraining or new
victim update collection. Private originals are evaluation-only attack truth.

paper_assets/build_comparison_panels.py has INDICES=[0,1]. Both sparse and
quantization panels use the same original first batch: Planner's read-only tensor
comparison found exact torch.equal for all16 originals and displayed examples.
Displayed global_index0 has dataset_index21950 / ILSVRC2012_val_00001048.JPEG;
global_index1 has dataset_index31051 / ILSVRC2012_val_00001433.JPEG.
These two are fixed illustration indices, not the full experimental dataset.
No choosing examples or image subsets using reconstruction quality.

Planner re-hashed the five input files and matched every saved manifest hash:
batch00: 24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141
batch01: 2f8db846205dda931162ca171282b8372a41bf8d3061c66bb6c83f65c7f20088
batch02: 1f1c898c93b7e0fcf4c00fecbb88fce2f913b4d4ed298aa21479a851c3284db3
batch03: ebdbd1774a320767a40f22c698260b4e3a9ac7ad4b69c2a92b987fe4318d570a
batch04: 6c30802ecdeb7c679e28f47b00e59f016464a3e850cca23fd5a396448ceae20f

Freeze a task-local registration including all80 metadata identities, batch
mapping, checkpoint/packet/input/source hashes, metric convention and seeds
before production. Check the same original identities across all conditions.

## Victim, observation and existing naive comparison

Preserve the protocol recorded in
quantization_aware_cpa/configs/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k.json
and quantization_aware_cpa/docs/RANDOM_FILL_SAME80.md, changing only keep fraction
and the necessary task/config/output names and support for those fractions.
This is the historical same80 track, not the separate newer quantization runner.

VGG16 IMAGENET1K_V1 round0; frozen conv, all classifier parameters updated; B16;
10 full-batch SGD steps, lr0.001, momentum0, weight_decay0, classifier dropout
disabled, model reset for each batch. Attack classifier.0.weight [4096,25088].
Historical packet convention is Y=(W_initial-W_after10)/0.001. Preserve it.
Do not silently replace it with signed raw theta_after-theta_start.

Reuse the actual existing received observation.bin for each fraction/batch at
results/cpa_fia_topk{50,30,20}_same80_B16/batch00..04/observation.bin (paths
relative to quantization_aware_cpa). All15 packets and completed matching naive
summaries exist. Their all80 identity tables are equal. Verify packet hashes
against the saved baseline summaries before use; never alter baseline files.

Global magnitude Top-k within FC1 only, floor(N*keep/100), N=102760448,
ascending flat index cutoff ties, row-major little-bit-order bitmap, retained
FP32 values unchanged; no quantization, rescaling or error feedback.
Actual packet bytes per batch: keep50=218365984, keep30=136157624,
keep20=95053444. Report actual retained count and bytes, not nominal keep percent
as bandwidth savings. Random filling adds zero transmitted bytes.

Use completed matched naive zero-fill metrics/reconstructions as the comparison.
Do not rerun the completed baseline merely to copy its output. Old B24/T1/R8
CoRA50/30 results are not this protocol. Keep10 remains its completed reference;
there is no fourth new fraction or extra baseline-restart control in this task.

## Fixed random-fill attack and observable selection

For each of five batches and each keep fraction, create20 independent fills.
Only missing coordinates are sampled once from U(-tau,tau), tau=min(abs(retained
FP32 values)), preserving transmitted FP32 bits exactly. Sampler
uniform_tau_pcg64_v1; chunk1048576; clip rounded endpoints to nextafter(tau,0).
Fill seed=100000+100*batch_index+candidate_index. Reuse this seed plan across
fractions. CPA seed=baseline seed0+batch_index for every candidate; FIA seed42.

Run the existing upstream CPA adapter, 25000 Adam iterations, lr0.001, no
scheduler, decor5.3, temperature7.7, ne1, nv0.13, l1=5, tv0.1, log_every5000.
Preserve its actual source behavior, including any upstream quirks; no FastICA,
extra gradient ICA, LS_A, debias, MP truncation, Stage2, oracle fills, source
scaling or true-source replacement. Input to FIA is abs(S_hat), estimated order.

FIA Adam lr0.1, TV0.1, FI1, cosine horizon25000 from the very first step.
Screen all20 candidates to5000. Rank by relative L2 between the received decoded
packet and the recompressed 10-step public SGD replay update from each whole16
reconstruction. Labels are public round0 model argmax; no true labels or images.
Use that condition's same Top-k convention, minimize score, candidate-index ties.
Advance top3 by continuing their saved pixels/Adam/scheduler/RNG states for20000
additional steps (total25000). Select one whole16 candidate among the three
finalists by the same observable score. No averaging, oracle best-of-K, per-image
selection, metric-based selection or altered ranking rule.
Freeze screen and final selections before loading private evaluation truth.

## Isolated implementation and verification

Existing run_random_fill_same80.py supports only10. Do not merely unlock its
constant: its baseline validator excludes50 and its report hardcodes Top-k10.
Create task-specific sibling runner/report/config files instead of editing old
hash-frozen code/configs, source algorithms or completed results. Suggested modules:
qacpa/run_paper_sparse_same80.py and qacpa/report_paper_sparse_same80.py,
plus scripts/run_paper_sparse_same80.py as a serial persistent wrapper.

Create three configs cpa_fia_random_topk{50,30,20}_same80_R20_S5k_K3_F25k.json
and new lock files, with outputs results/cpa_fia_random_topk{keep}_same80_R20_S5k_K3_F25k.
Whitelist exactly the three newly authorized fractions in this task wrapper.
Reuse verified unchanged random_fill.py/staged_fia.py/upstream_naive.py functions
where possible. Never mutate old module globals as a shortcut around validation.

Special case: old50 config has no keep_percent field and uses codec label
topk50_global_bitmap_fp32_v1; 30/20 use topk_global_bitmap_fp32_v1. Both source
codecs use magic QACTOPK1 and the same32-byte layout, but verify their 50% mask,
FP32 value bits, cutoff ties and decoded tensors/serialized bytes agree before
using the generic reader/recompressor for50. Preserve the original received
packet bytes and record its actual source codec label. Make the new validator
understand this explicit historical schema difference without rewriting history.

Before production, verify same80 hashes/order, all15 original packet hashes,
retained bits/threshold limits/seed determinism, percentage-correct replay and
selection, no private data in attack/ranking, and exact staged continuation.
Use focused synthetic/contract checks and existing meaningful tests; no extra
real-image optimization pilot, changed objective or parameter search.
Verify attack dependency hashes against the frozen10/baseline implementations.
On an unexplained mismatch, report it; do not suppress provenance checks or edit
saved manifests to make them pass. Authenticated old snapshots may be used in an
isolated package with exact hashes; do not overwrite current research sources.

Example final runner interface from quantization_aware_cpa:
C:/miniconda/envs/DLG/python.exe -u -m qacpa.run_paper_sparse_same80 --config configs/cpa_fia_random_topk50_same80_R20_S5k_K3_F25k.json --phase run
Repeat serially for30 then20 through the persistent wrapper, without starting
concurrent GPU workers. Record actual implemented commands in the report.

## Evaluation and paper-ready deliverables

After final selection, preserve the existing signed-source absolute-cosine
Hungarian correspondence within each original16, then apply the SAME assignment
to all image metrics/figures. Use saved private Z only in this offline matching;
no truth scaling or replacement. Evaluate all80 with existing PSNR, SSIM and
VGG LPIPS implementations/ranges. Report quality changes, not success rates.
Do not add success thresholds, compromise probabilities or a new 4% claim.

Per fraction save summary.json, report.md, per_image_metrics.csv,
per_batch_metrics.csv, comparison_metrics.csv, all80_rec.pt in original order,
all candidate screen/final outputs, selections/scores/seeds, packet audits,
source/input manifests, native reconstructions and original-vs-naive-vs-random
contact sheets. Keep intermediate states sufficient for authenticated resume.

In SWEEP_OUTPUT_ROOT save combined_comparison_metrics.csv with all three paired
naive/random conditions, a manifest linking unchanged10 for later paper use,
fixed-index0/1 comparison images, and all80 comparison images. No enhancement,
quality-based example selection or success-color borders. Export originals and
saved reconstructions without recomputing matching during plotting.
The compute budgets differ between naive and random: disclose this when reporting
differences; do not attribute an improvement solely to missing-value imputation.

Prepare outputs for the existing paper_assets renderers; leave shared manuscript
and paper_assets files to Planner's result review so incomplete results never
replace N/A. Preserve test.tex formatting, existing panel indices and title-free
layout. Required final report also lists actual source paths and validation.

## Bounded compute, launch and completion

New budget exactly3 fractions x5 batches x20 CPA candidates =300 CPA trajectories
x25000=7500000 CPA iterations;300 screen FIA x5000 plus45 finalists x20000=
2400000 FIA iterations;345 public-ranking replays. No new victim training,
additional seeds, fraction search, naive restarts or extra attack arms.

Historical10 measured67393.72 seconds compute (~18.72h) for one fraction on the
saved run. Three conditions at that rate are ~56h, an estimate only. Set a
task-wide96h active-compute ceiling, <=14GiB allocated CUDA memory and <=160GiB
new disk usage; free workspace disk at planning was228.42GiB. Run serially, do
not persist dense filled FC1 matrices or duplicate whole source datasets.
Verify estimated storage before launch and log actual time/memory/storage.
Resource/numerical failures must be reported without dropping images, changing
the method or exceeding this budget. Same-scope implementation fixes and resume
from authenticated completed/checkpoint stages are authorized; do not silently
restart completed trajectories or launch extra trials after seeing results.

Once gates pass, write CLAIMED_BY and RUNNING to the active handoff. Launch one
hidden persistent serial wrapper; verify PID/start and one startup receipt.
Register one model-free process-exit callback through .ai/completion_notify.py
to this existing Executor, then end the model turn. No model polling loop or
monitoring agent. On exit finish authorized evaluation/reporting, record actual
COMPLETED/FAILED/NEEDS_REVIEW, and send one completion notice to the exact current
Planner thread above only if supported and authenticated; otherwise save the
report and state that notice delivery was unavailable. Historical Planner IDs
are not valid substitutes. QUEUED is distinct from claimed/running/completed.
