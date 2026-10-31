# Original CPA / CIFAR-100 / deterministic per-tensor bits 8--4

TASK_ID: CPA_CIFAR100_ORIGINAL_CPA_PERTENSOR_BITS87654_20260920
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: USER_AUTHORIZED_ORIGINAL_CPA_CIFAR100_BIT_SWEEP
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_ORIGINAL_CPA_CIFAR100_PER_TENSOR_BITS8_TO4_AND_LUNA_DELIVERY_20260920
PLANNER_THREAD: 01a0b9f1-ab68-7743-a9bc-757c5cce7ccc
EXECUTOR_TARGET: USER_DESIGNATED_EXISTING_TERMINAL_LUNA
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/cifar100_original_cpa_pertensor_bits87654_B16_T10_20260920
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/cifar100_original_cpa_pertensor_bits87654_B16_T10_20260920
RESULT_REPORT_PATH: .ai/execution-results/CPA_CIFAR100_ORIGINAL_CPA_PERTENSOR_BITS87654_20260920/report.md

## Authorization and roles

The user requested:
- "기존에 하던거 fast ica + gd ica 하지말고 CPA 원문 코드로 갈아낀다음에 cifar100 version으로 8~4bit 실험 뽑아줘"
- "PER TENSOR로"
- "여기가 아스트라 xhigh고 다른 세션에 루나가 켜져있어 수정해주고 실행은 거기로 넘겨"
- "별도로 연 터미널 Luna"

This authorizes necessary isolated implementation, meaningful checks, CIFAR-100
acquisition, public FC2 training if no compatible checkpoint exists, client update
collection, the five requested bit conditions, offline evaluation and reporting.
It does not require another scientific execution approval. Sandbox/network
permissions still apply through their normal mechanism.

Current Planner is Astra/xhigh PID3348. Executor is the other existing terminal
Luna, NOT the VS Code Luna conversation. See SESSION_ASSIGNMENT.md. At preparation,
terminal PID18908/start1789826050208ms has startup thread
01a0b9f2-29b8-7a03-8a3b-733ed36ece8b but no persisted first conversation or verified
runtime yet. Verify its actual saved thread/model/effort after first input before
claiming. Do not invent a model switch, create another agent, or claim queue
acceptance means optimization has started. Inherited executor effort is xhigh;
if the user's actual Luna uses a different effort, report that operational
difference rather than silently changing global settings.

## Prior task ownership and preserved evidence

The prior row-wise handoff was inconsistent: header READY/unclaimed, footer RUNNING
under historical thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab. Its expected public
output directory was absent, and process inspection found no standalone CPA
research worker; existing Python processes were Jupyter helpers, nvitop and the
message broker. No prior completion is inferred.

The user explicitly replaced FastICA+GD work with the new request. The old handoff
is preserved byte-for-byte at
.ai/execution-handoffs/CPA_SIGNED_INT4_ROWWISE_20260919_before_cifar100_20260920.md
SHA25641f9d680ce5c99ecc6074669ec77fb99492d56ce3ada3dd1758f4aefa5f75b5c.
Its old contradictory claim remains historical evidence. Before launching, check
once for conflicting live research ownership. Preserve old results and source.
Do not resume the superseded row-wise experiment or terminate unrelated processes.

## Scientific scope and corrections to the preliminary relay

User-authorized changes are original CPA separator, CIFAR-100/compatible FC2,
per-tensor granularity, and bits {8,7,6,5,4}. Preserve the existing single-client
FedAvg B16/T10, local SGD lr0.001, and signed raw delta theta_after - theta_start.
Turn63's FedSGD and B256 suggestions were not user-specified and are NOT inherited.
Turn64 correctly establishes that B need only be <= hidden width; B=16 is valid
for h=256. This also resolves the B256 boundary concern without changing the
inherited client size. T10 means TEN full-batch SGD optimizer steps on the same
16 client images; it is unrelated to CPA temperature or number of test batches.

Planner choice within the requested experiment: ten independent test batches of
16 images, sampled once reproducibly without overlap; all five precisions share
the same 160 identities, initial model and ten raw client updates. These are ten
independent clients/round-zero trials reset to the same starting model, not ten
sequential global FL rounds. Use seed42 for split/order/model training, and record
all identities, local update rules and reproducibility evidence before attack.

This is a new FC2/CIFAR-100 track. It is not an isolated separator ablation against
the old VGG16/ImageNet results; dataset, architecture and image resolution differ.
No claim of progress on VGG16 end-to-end FIA follows from this track.

## Model, data and original source

Use FC2 from official_cpa/src/models.py: 3072 -> 256 -> 100, ReLU, attack_index0.
Dataset normalization is the original CIFAR-100 mean/std; test inputs are 3x32x32
with no random augmentation. Train model on CIFAR-100 TRAIN data only; client
updates use TEST images. Labels may be used by model/client training only.

Use a compatible pre-existing public FC2 checkpoint only if provenance/settings
can be authenticated. Otherwise train exactly once with the original train.py
defaults: 10 epochs, batch128, Adam lr0.001, weight_decay0.0005, no scheduler,
h_dim256, seed42. Freeze the final epoch checkpoint before creating bit arms;
no best-test checkpoint selection or accuracy-triggered extension.

Keep official_cpa source unchanged. At planning its Git HEAD is
e2dc6ddb1a1bd262fcc36ccf2bcd837a972cc25a, and gradient_inversion.py has no local diff.
Hash actual imported source and record all existing modifications before use.
Do not assume directory name alone proves upstream identity. In particular,
CocktailPartyAttack.step currently resets computed loss_ne to zero; preserve
original behavior and disclose it, rather than silently repairing the algorithm.

Wrap original FC2/CocktailPartyAttack and use isolated output/cwd/module path
adapters for Windows, offline logging and public/private separation. Preserve
original whitening, identity initialization, loss formulas and get_rec behavior.
No edits to old codecs, prior experiment runners or official_cpa algorithm files.

## Quantization and attack input

All FOUR parameter tensors are sent:
net.0.weight [256,3072], net.0.bias [256],
final.weight [100,256], final.bias [100].
One FP32 scalar scale per entire tensor, including each whole bias vector.

For b in 8,7,6,5,4:
qmax = 2**(b-1)-1;
m = max(abs(delta)) in CPU FP32;
s = float32(m/qmax);
q = round_nearest_even(float32(delta/s)).clamp(-qmax,qmax);
decoded = q.float()*s.
For an exactly zero tensor use q=0,s=1. Reject nonfinite input or nonzero-scale
underflow; do not clip or perturb the truth to satisfy a test. Serialize integer
codes plus scale/shape/codec metadata; re-read and validate the actual written
observation. Each bit is derived directly from the SAME raw FP32 update.

Do not use legacy qacpa/quantizer.py: its observation projection depends on private
truth and can move outputs off the quantization grid. Existing signed scalar
codec primitives may be reused only if their contract supports FC2 arbitrary
tensor shapes without importing VGG16 constants.

CPA consumes only decoded net.0.weight [256,3072], in raw signed update units.
Other transmitted tensors affect communication accounting but not this CPA
separator input. Attack must not load raw updates, private images or labels.
Provide labels=None/an inert unused value only if the original CP class signature
requires it; use_labels remains false. Keep all private truth in separate files.

This registered sweep is the naive cell-center original-CPA baseline: exactly
one center input per bit and batch. There are no random members, FastICA,
additional gradient-ICA refinement, FIA, learned priors or candidate screening.
An FP32 update is collected by the sender but no additional FP32 attack arm or
hyperparameter sweep is included in this five-bit execution budget.

## Fixed CPA settings

Run original CocktailPartyAttack for 25000 Adam steps, lr0.001, scheduler none,
n_comp16. Adopt the original README FC2 example's fixed settings:
decor1.47, tv3.1, temperature T12.4, ne1, l1=0, nv=0.
This is an explicitly declared transfer of Tiny-ImageNet example hyperparameters
to CIFAR-100, not a claim that CIFAR-100 settings were published or validated.
It implements Turn64's no-extra-tuning alternative. Default tv0 is not used.
Register this before reading any reconstruction metrics; no outcome-based tuning.

Use seed42 consistently for each bit's corresponding attack. Original identity
initialization is preserved. Give every batch/bit its own whitening cache and
checkpoint directory; stale cache reuse across bits must be impossible. CUDA
FP32 is preferred in the existing DLG environment. Record actual source hashes,
device, dtype, seeds, iteration counts, timing, memory and numerical failures.

## Implementation, evaluation and deliverables

Create isolated qacpa/cifar100_original_cpa_*.py modules or an isolated package,
focused tests and a persistent task-local wrapper. Before production:
- validate actual wire roundtrip, signs, ties, zero tensors, all bit ranges,
  whole-tensor/bias scaling, no raw truth in public files and exact q*s input;
- validate FC2 shapes, B16 support, ten actual SGD steps, same batch/checkpoint
  across arms, original CPA class invocation/config and separate cache paths;
- compare unchanged upstream function/source identity, and synthetic adapter
  parity. No real-image optimization pilot or extra seed is needed.

Freeze public outputs before offline metrics. A separate evaluator may load
private x, but labels/raw updates must not guide attack, matching selection or
hyperparameter choice. Evaluate ALL 16 images per batch using the original
evaluation conventions; disclose any evaluation-only sign/permutation alignment,
save its mapping and keep the public raw reconstruction unchanged. Use the SAME
assignment for LPIPS/PSNR/SSIM/figures, fixed before comparisons between metrics.
Record metric version/backbone/range and any 32x32 resizing required by that metric;
do not resize attack inputs or mislabel upscaled display as native reconstruction.

Write per-image and per-batch CSV, all-image means, batch medians/IQR, and paired
4bit-minus-8bit batch differences. Do not claim statistical significance from the
reviewer's heuristic of mean difference exceeding IQR; report descriptive paired
effects without a new success cutoff. Explain reconstruction failures faithfully.

Export native saved reconstructions, original-vs-all-five-bit contact sheets
covering all160 images, a readable overview, and a ZIP with manifest/hash/index.
Do not cherry-pick good reconstructions. Report logical b-bit value bytes
separately from actual physical storage (e.g. int8 arrays are not packed b-bit),
four scales = 16B per packet, headers/metadata and exact total file/transport sizes.

Required outputs: report.md, execution_summary.json, immutable registration,
dataset/checkpoint/update/packet/source manifests, public final audit, timing and
memory records, raw checkpoints/reconstructions, private metrics CSV/JSON,
contact-sheet PNGs and archive. Save metrics before plotting so export repair
does not require repeating optimization. Use C:/miniconda/envs/DLG/python.exe.

## Budget, launch and completion

Budget: one public model training (only if needed), ten B16/T10 client updates,
five precisions x ten batches = 50 CPA runs x25000 steps = 1250000 CPA steps.
Planning safety ceilings: 24 hours real computation cumulatively, CUDA allocated
<=12GiB, new output storage <=20GiB excluding dataset/runtime packages.
These are ceilings, not runtime forecasts. Download time is logged separately.
Stop and report an actual resource/numerical failure rather than change method,
silently drop a bit/batch, extend budget or restart completed trajectories.

Check data/dependency availability first; download CIFAR-100 and required metric
weights only as needed using normal network permission mechanisms. W&B/external
telemetry remains disabled. Same-scope implementation repairs and authenticated
checkpoint resume are authorized; additional scientific arms/tuning are not.

Once all gates pass, launch hidden persistent serial execution, verify PID/start
and one startup receipt, register one model-free OS process-exit notification to
THIS existing Luna, and end its model turn. Do not repeatedly poll. On exit finish
authorized evaluation/reporting, write actual completion/failure state and send
one result notice to current Planner thread01a0b9f1-ab68-7743-a9bc-757c5cce7ccc.
Do not use historical Planner IDs. The Planner ends after one delivery receipt;
QUEUED, CLAIMED/RUNNING and completed scientific results are distinct states.
