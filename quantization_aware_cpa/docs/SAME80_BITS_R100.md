# Same80 signed quantized CoRA, bits 8–4

This isolated run compares a Naive observed-code center with a public-selected
CoRA reconstruction for the same 80 historical ImageNet images. It preserves the
five original independent B16 batches and uses the cached `Y_delta_over_lr`
observation from the existing M6 results. It performs no new client training,
local update collection, or B80 regrouping.

The fixed configuration is
`configs/same80_bits87654_R100_S5k_K3_F25k.json`. It is validated as an exact
contract by both the runner and persistent wrapper. The attack imports only the
public int8 code packets, their observed FP32 scales, and the registered VGG16
checkpoint. Preparation authenticates the historical caches and writes those
packets in a separate process. Private images, labels, evaluation features, and
raw FP32 observations are opened only by the separate report process after the
public gate passes.

## Frozen protocol

- Bits are processed in order 8, 7, 6, 5, 4; each bit has batches 00–04.
- Each batch is quantized independently with signed symmetric per-tensor FP32
  absmax scaling and nearest-even rounding. Code storage uses int8; logical bit
  width is recorded separately. The output makes no packed-bandwidth claim.
- Each public packet produces 100 reproducible bounded-FP32 candidates with the
  registered extremal anchor, seed formula, and 32-row sampling chunks. Every
  candidate must reproduce the exact codes and scale.
- Original upstream CocktailPartyAttack runs for 25,000 Adam steps per candidate
  and per Naive center. It receives the observed matrix directly and keeps the
  approved batch-index seed and objective settings.
- Original Direct FIA receives `abs(S_hat)` in estimated source order. All 100
  candidates run the first 5,000 steps with a fixed 25,000-step cosine horizon.
  The three lowest finite public scores continue their saved image, optimizer,
  scheduler and RNG state for 20,000 more steps. Public ties break by candidate
  ID ascending. The entire 16-image candidate is selected at 25,000 steps.
- Selection uses mean feature cosine distance plus `0.1 × normalized TV`. It
  never reads private values. Hungarian correspondence is applied only after
  the public selection is frozen, using absolute source cosine against
  `Z_evaluation_only`; the same permutation supplies PSNR, SSIM and VGG LPIPS.

## Commands and restart behavior

Run each phase from this directory with the DLG environment:

```powershell
C:/miniconda/envs/DLG/python.exe -m qacpa.run_same80_bits_r100 --config configs/same80_bits87654_R100_S5k_K3_F25k.json --phase prepare
C:/miniconda/envs/DLG/python.exe -m qacpa.run_same80_bits_r100 --config configs/same80_bits87654_R100_S5k_K3_F25k.json --phase run
C:/miniconda/envs/DLG/python.exe -m qacpa.run_same80_bits_r100 --config configs/same80_bits87654_R100_S5k_K3_F25k.json --phase gate
C:/miniconda/envs/DLG/python.exe -m qacpa.report_same80_bits_r100 --config configs/same80_bits87654_R100_S5k_K3_F25k.json
```

The persistent `scripts/run_same80_bits_r100.py` wrapper invokes those phases
sequentially, stores per-phase logs and stops on the first nonzero exit. A
process lock prevents duplicate wrappers and a separate public-root lock
protects attack stages. Completion markers bind artifact hashes to the task,
configuration and code identities. Checkpoints without a valid integrity
sidecar are quarantined after an interrupted attempt and that stage restarts
with the same seed. Recorded numerical or general failures are not retried
automatically. Cleanup is restricted to named disposable caches beneath this
task's public root.

The first scheduled Naive center (bits 8, batch 00) supplies the only resource
measurement; it is not an extra pilot. The record includes CPA/FIA wall times,
CUDA peaks, disk free space, remaining step counts and a linear wall-time
projection with 50% uncertainty. Disk checks retain the configured 8 GiB
reserve. Resource failure stops the wrapper for review without changing any
scientific condition.

## Outputs

Public artifacts are written under
`results/same80_bits87654_R100_S5k_K3_F25k`. Each bit/batch cell stores the
registered packet, members `000`–`099` with their sample/CPA/source records and all 5,000-step reconstruction
images and scores, full ranking history, selected continuation checkpoints,
three 25,000-step endpoints, the distinct Naive endpoint, and a public gate.
Dense sampled FC1 matrices and nonselected continuation checkpoints are
temporary; all source estimates and required image outputs remain available.

Private evaluation outputs are written under
`results_private/same80_bits87654_R100_S5k_K3_F25k/evaluation/bitNN`. Each bit
contains per-image and five-batch metrics, the all-80 original/Naive/CoRA
comparison PNG plus one comparison PNG for each original batch, ordered
reconstruction tensors, public selection histories, a machine-readable summary
and `report.md`.

The search budgets differ: CoRA searches 100 randomized candidates and publicly
selects one whole 16-image reconstruction per batch; Naive uses one center
candidate. Metrics at 5,000 steps remain separate from the 25,000-step endpoint
metrics. No per-image best candidate or historical success cutoff is introduced.
