# Full VGG16 single-client update inversion

This is direct image optimization without CPA source separation. Both the
quantization-cell method and the decoded-center comparison use the same full
Conv+FC local training replay. Existing FC-only observations and experiments
remain separate; they cannot supply missing Conv updates.

## Public observation and private evaluation

The user confirmed on 2026-09-13 that the attacker receives the packed 4-bit
codes and FP32 row radii, the starting VGG16 checkpoint, and the local training
rules (SGD, learning rate, steps, and ordered minibatches). The checkpoint and
contract are bound to the observation by SHA256. True images, true full-precision
updates, original labels, and the sender's quantization RNG seed are private.

The packet represents raw `final_parameter - initial_parameter`. Conv weights
use `[output_channels, -1]`, FC weights their existing matrix shape, and each
bias tensor uses one `[1, numel]` row. Giving each scalar bias its own radius
would reveal its absolute value, so that layout is rejected. Every weight and
bias must occur exactly once, with the expected VGG16 name and original shape.

The 4-bit claim applies to codes per value. Report code bytes, radius bytes,
packet header bytes, and serialized manifest bytes separately. The receiver API
does not accept an evaluation-truth path or an FP32 target-update argument.
Only `collect` and the separate offline `evaluate` command read private data.

## Registered initial run

`configs/full_update_single_client_B16_T10.json` retains the existing same80
inputs, B16, T10, SGD learning rate 0.001, momentum/weight decay zero, disabled
dropout, and ImageNet normalization. All Conv+FC parameters train. Batch indices
are explicit for every step. The initial pilot has three outer iterations,
seed 42, Adam image/label learning rates 0.1, TV weight 0.0001, and radius weight
1. These are implementation pilot settings, not established optimal settings
or a sufficient budget for image recovery. `probe` performs one gradient and
does not take an Adam step or emit a reconstructed image.

`..._recompute.json` has the same contract and pilot parameters; only its
gradient strategy changes. Recompute regenerates each parameter state from the
public starting model, then evaluates the exact reverse SGD derivative for all
local steps. It does not truncate gradients or shrink model/batch/step count.
It currently rejects nonzero momentum. T10 needs 55 ordinary local steps plus
10 reverse Hessian-vector steps instead of storing the full trajectory.

The CLI disables TF32, requests deterministic cuDNN and PyTorch algorithms,
and sets the CUDA deterministic workspace configuration. The CPU module tests
check replay against actual PyTorch SGD, image derivatives against finite
differences, and reverse recomputation against full unrolling, including label
probabilities and the pixel/softmax chain rules. Small-model tests establish
implementation correctness; they do not establish full-VGG feasibility.

At fixed 224px input, VGG16 produces a 7x7 feature map and its adaptive 7x7
average pooling is exactly identity. The loader uses a checked identity module
for this case because the general CUDA adaptive-pooling backward rejects strict
determinism. The replacement rejects other feature sizes. Exact forward,
first-derivative, and second-derivative equivalence are covered by tests.

## Loss and final checks

For a positive row radius `r`, the coordinate is `7.5 * (delta/r + 1)`.
Stochastic code `q` gives the closed support `[q-1, q+1]` clipped to `[0,15]`.
The cell loss penalizes violations, while the center baseline penalizes distance
to `q`. Both add row-radius agreement with `max(abs(candidate_delta))` and use
equal tensor weighting after tensor means. Zero-radius rows use an explicit
absolute update-unit convention because no quantization step exists there.
Double arithmetic protects small radii; row chunks bound temporary storage.

Private labels are optimized as a softmax relaxation. The final candidate is
replayed with hard argmax labels and checked again. Closed-support fit, zero
marginal probability at boundary points, exact joint RNG matching, and image
recovery are different claims. Exact RNG matching is not checked. The attack
saves its final candidate without truth-based checkpoint selection.

Offline evaluation uses one pixel-MSE Hungarian assignment for all samples and
all reported MSE/PSNR/SSIM/label metrics. It never sends the assignment back to
the attack. LPIPS and a scientific recovery-success threshold are not selected
implicitly. A short pilot cannot establish comparative recovery performance.

## Commands (PowerShell, from quantization_aware_cpa)

```powershell
$fullPython = 'C:/miniconda/envs/DLG/python.exe'
$fullWeights = 'C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth'
$fullConfig = 'configs/full_update_single_client_B16_T10_recompute.json'
& $fullPython -B -m qacpa.run_full_update inspect --config $fullConfig --checkpoint $fullWeights

# Sender only. Choose fresh output directories; existing paths are rejected.
& $fullPython -B -m qacpa.run_full_update collect --config $fullConfig --checkpoint $fullWeights --inputs results/m6_uia_fp32_B16_fia/batch00/inputs.pt --output results/full_update_new/batch00/observation --private-output results_private/full_update_new/batch00 --codec-seed 0

# Receiver: no private image, label, update, or sender seed argument.
& $fullPython -B -m qacpa.run_full_update probe --config $fullConfig --checkpoint $fullWeights --observation results/full_update_new/batch00/observation --output results/full_update_new/batch00/probe --mode cell
& $fullPython -B -m qacpa.run_full_update attack --config $fullConfig --checkpoint $fullWeights --observation results/full_update_new/batch00/observation --output results/full_update_new/batch00/cell --mode cell
& $fullPython -B -m qacpa.run_full_update attack --config $fullConfig --checkpoint $fullWeights --observation results/full_update_new/batch00/observation --output results/full_update_new/batch00/center --mode center

# Offline evaluation only, after final candidates are fixed.
& $fullPython -B -m qacpa.run_full_update evaluate --candidate results/full_update_new/batch00/cell/candidate.pt --truth results_private/full_update_new/batch00/evaluation_truth.pt --output results/full_update_new/batch00/cell/evaluation.json
```

Resource failures are recorded without automatically changing fixed conditions.
Check one full outer gradient and its time before expanding the iteration budget.
CUDA allocator failures may precede the Python exception on Windows; externally
stopping such a run must be recorded as a stop after observed allocation failures,
not as a completed gradient or an exact peak-memory measurement.

## First measured resource result (2026-09-13)

The new batch00 observation contains 138,357,544 parameter values across 32
tensors: 69,178,772 code bytes, 53,728 radius bytes, 1,024 packet header bytes,
and 12,206 manifest bytes. New full Conv+FC collection completed all 10 local
steps. This is measured packet accounting, not a total 4-bit-per-value claim.

The initial full-unroll implementation hit repeated CUDA allocator failures
and was stopped after approximately 130 seconds without completing an outer
gradient. A subsequent strict-determinism attempt failed at the general
adaptive-pooling backward; the checked identity implementation resolved it.

Reverse recomputation with row-loss checkpointing completed the actual
full-VGG16 B16/T10 cell-loss outer gradient in 15.512 seconds. Peak PyTorch
allocated memory was 5,936,897,024 bytes (5.53 GiB), and reserved memory was
8,050,966,528 bytes (7.50 GiB). The image gradient was finite and nonzero.
This verifies one iteration's resource feasibility. These implementations are
recorded separately; the failed early unroll is not a controlled performance
comparison against every possible unroll implementation.

The separately recorded pilot budget is three outer iterations per arm, with
a projected optimization limit of 600 seconds per arm. The measured cell probe
projects 46.54 seconds for three gradient computations, excluding initialization
and final hard-label validation. It does not predict sufficient iterations for
recovery. See `results/full_update_single_client_B16_T10/resource_plan.json` and
the per-run registered designs/results for observations.

The final three-iteration candidates both failed hard-label observation closure
fit. Cell/center PSNR were 7.48790/7.48856 dB, SSIM 0.008083/0.008089, and both
had label accuracy zero. Times including final hard-label validation were
47.79/48.05 seconds. These short pilots establish neither successful recovery
nor a method ranking. The registered designs, implementation hashes, checkpoint,
observed packet, and all attack settings match between arms except loss mode.

The separate private sender diagnostic regenerated all ten local prefixes and
compared their raw parameter deltas to ordinary `torch.optim.SGD`. All 32
tensors matched exactly at every prefix, with zero absolute error and zero
error relative to transmitted quantization steps. This checks forward replay,
not the inversion's image correctness. The report stays beside the private
truth file. Run it explicitly with `python -B -m qacpa.validate_full_update_replay`
and `--config`, `--observation`, `--checkpoint`, `--truth`, `--output`; its output
must be inside the isolated truth directory. The receiver never imports it.

All 281 targeted codec, observation, replay, adjoint, pipeline, CLI, evaluation,
pooling, and private-diagnostic tests passed. Machine-readable measured results
are in `results/full_update_single_client_B16_T10/summary.json`.

## User-requested iteration increase

The user subsequently requested more iterations. The new config
`configs/full_update_single_client_B16_T10_recompute_I100.json` changes only
`attack.iterations` from 3 to 100. This is 100 image-optimization iterations;
local training stays B16/T10. Each arm starts from the same seed42 initialization
as its pilot, since the pilot candidate does not contain Adam state for resume.
The first three logged numerical values of both new runs match their pilots
exactly. Both arms retained their final post-update candidate and used the
same separate hard-label and offline evaluations.

The completed outputs are `batch00/attack_cell_I100` and
`batch00/attack_center_I100` under the same result root. The fresh experiment
plan is `batch00/I100_plan.json`; its time estimate is approximately52 minutes
for both arms. That estimate and an in-progress loss decrease are not final
recovery results. `batch00/I100_start_validation.json` records the initial
provenance checks and additional current codec/CLI hashes; those additional
hashes were not previously recorded by the pilot.

After both 100-iteration final candidates and their offline evaluations exist,
the standalone reporter validates the recorded fingerprints, exact three-step
pilot prefixes, and existing final evaluations before writing a new comparison:

```powershell
& $fullPython -B -m qacpa.report_full_update_iterations --base results/full_update_single_client_B16_T10/batch00 --truth results_private/full_update_single_client_B16_T10/batch00/evaluation_truth.pt --output results/full_update_single_client_B16_T10/iterations_100_comparison --image-grid --lpips
```

This offline-only report includes all 16 images using their existing pixel-MSE
Hungarian assignments. Supplemental VGG LPIPS uses FP32 RGB in [-1,1] and the
same assignments, with locally verified backbone/calibration weights. It does
not select candidates or change the attack. No LPIPS success cutoff is applied.
The interpretation amendment was recorded during optimization, before either
100-step final candidate or private evaluation; intermediate soft losses had
already been observed. Single-batch numerical differences are descriptive and
do not establish a general performance ranking. A negative result at this
budget does not prove inversion impossible. Sixteen focused reporter tests
passed, including a small CPU inference with the actual cached LPIPS weights.

## Completed 100-iteration comparison (2026-09-14)

Both registered runs completed all 100 iterations. The following means include
all 16 images, using each final candidate's fixed offline Hungarian matching.
Higher PSNR/SSIM and lower LPIPS correspond to better image agreement.

| Mode | Iterations | PSNR dB | SSIM | VGG LPIPS | Hard-label bin violation rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| cell | 3 | 7.48790 | 0.008083 | 0.839900 | 22.4134% |
| cell | 100 | 6.08976 | 0.005472 | 0.864648 | 25.3444% |
| center | 3 | 7.48856 | 0.008089 | 0.839646 | 22.7115% |
| center | 100 | 6.02765 | 0.005649 | 0.871043 | 21.3540% |

Label accuracy is zero in every row. All 32 parameter tensors fail the existing
hard-label closure tolerance in both 100-iteration candidates. The elementwise
bin violation rates above concern positive-radius coordinates; the full
closure check also includes row-radius and zero-radius errors. Center's rate
decreased while its image metrics worsened, so observation fit and image
agreement must remain distinct measurements.

The final prestep soft objectives are 9,558.68 (cell) and 9,005.29 (center),
down from their respective initial objectives near16.9 million. This did not
translate into better PSNR, SSIM, or LPIPS. Final hard-label diagnostics are
poststep values. Final soft logits were not saved, so these logs and artifacts
alone cannot isolate a label-relaxation gap or establish the cause of poorer
image agreement. The all-image grid shows noise-like candidates. These are
budget-limited results for one batch and seed, not an impossibility result.

Wall times including final hard-label validation were1502.95 and1501.64 seconds
(3004.59 seconds combined, about50.08 minutes). Both peaked at5,955,634,176 bytes
of CUDA allocated memory (5.55 GiB). Offline evaluation/report generation is
additional. No snapshot-path or component timing benchmark was performed.

The completed comparison, hashes, supplemental metric provenance, convergence
plot, PSNR plot, and all16-image grid are in
`results/full_update_single_client_B16_T10/iterations_100_comparison/`.
`batch00/I100_completion_validation.json` additionally verifies the unchanged
current implementation/config hashes and generated report artifact hashes.
Claude Turn19 accepted the comparison/reporting design and corrected the
unsupported LPIPS0.1 criterion. No image-success cutoff was adopted; defining
one for this full-update task remains a user decision for future experiments.
