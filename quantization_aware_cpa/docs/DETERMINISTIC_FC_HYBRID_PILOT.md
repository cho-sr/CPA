# Deterministic FC hybrid pilot

The user authorized a short experiment and explicitly froze Conv on 2026-09-14.
This pilot is separate from the historical stochastic Conv+FC100 experiment,
which did not contain a CPA/FIA feature term. Claude's actual review is RELAY
Turn25; Codex's independent response and updated user scope are Turn26.

The sender uses the same batch00 of16 images, VGG16 public starting checkpoint,
ImageNet normalization, and10 repeated full-batch FC SGD steps at LR0.001,
momentum0, weight decay0, dropout0. Conv does not train. The observed delta is
final-minus-initial for all six FC weight/bias tensors. Sender raw deltas must
match an independent replay exactly before publishing packets.

The new codec uses16 affine levels over each row's public FP32 absmax radius.
CPU FP64 computes `t=(value+radius)*7.5/radius`, clips to `[0,15]`, and uses
nearest-even rounding. Codes are packed two per byte; codec number2 is distinct
from stochastic1. Zero-radius rows use code0. Decoder output depends only on
code+radius; private-data-dependent floating point boundary correction is absent.
Positive-radius half-step cells and radius consistency are separate loss terms.
Final diagnostics report exact candidate reencoding and radius matching as well
as fixed-observation closure errors. The optimizer's positive scale floor never
changes the observed cells or final codebook.

The public receiver decodes FC1, converts `-delta/lr` using the public LR, and runs
the existing official CPA adapter for25,000 iterations at seed42. The fixed FIA
target is `abs(S_hat)` in estimated source order. It is a fallible estimate from
the same packet, not independent information or a known true feature. The
official implementation's configured negentropy weight is inactive because its
step replaces that loss with zero; TV is inactive when its nv branch is used.

Registered plan: `configs/deterministic_fc_hybrid_pilot.json`. A common500-step
FIA warmup begins from random RGB pixels with the exact public checkpoint's
frozen Conv extractor. Both arms start from the same postwarmup pixels and public
model label estimates, with fresh identical Adam image states. Each gets50
additional image steps. The hybrid adds bin+radius gradients to images and
private label logits; the FIA control retains its common initial label estimate.
Thus this tests the complete observation-loss addition, not bin alone or an
image-only effect with label estimation held fixed. Equal iterations and shared
LR schedule are used, not equal runtime or separately tuned best performance.

At the shared starting candidate, public gradient norms set the fixed TV weight
to10% of FIA gradient norm and bin+radius weight to25%. These are preregistered
pilot heuristics, not proven optimal values. Positive observation denominators
use `max(h,median_positive_public_h)`; zero rows use the same fixed public floor
for their separate absolute error. All FC rows remain represented.

The exact FC-only adjoint returns feature and probability gradients; the receiver
backpropagates through frozen Conv and the softmax Jacobian and adds direct FIA
and TV gradients. CPU tests compare the entire composition with full unrolling.
Final pixels, hard labels, soft logits, optimizer and RNG states are saved.
Final soft/hard fit is evaluated at the same saved image. Iteration losses and
gradients are prestep; actual displacement and clamping diagnostics are poststep.

Private evaluation is a separate process after both arms finish. All16 images
are evaluated with one pixel-MSE Hungarian assignment per arm, reused for PSNR,
SSIM and locally cached VGG LPIPS. Labels also get a multiset-overlap report.
No true image/feature/label supplies initialization, correspondence, early stop,
candidate selection or weight calibration. No LPIPS success cutoff is invented.
Mixed metric directions or this short pilot's failure do not establish a general
advantage, impossibility, or required iteration count.

Public run root: `results/deterministic_fc_hybrid_pilot_B16_T10`.
Private sender/evaluation root: `results_private/deterministic_fc_hybrid_pilot_B16_T10`.

Completed results (all16, no success threshold):

| Metric | CPA + FIA | CPA + FIA + bin/radius |
| --- | ---: | ---: |
| PSNR dB | 10.87442 | 10.85184 |
| SSIM | 0.129483 | 0.119128 |
| VGG LPIPS | 0.683371 | 0.684643 |
| Final soft observation objective | 19.24596 | 6.40084 |
| Final hard observation objective | 92.77442 | 108.17185 |
| Hard exact code mismatch rate | 35.6503% | 36.3161% |
| Hard positive-radius closure violation rate | 31.4379% | 32.0924% |
| Additional50-step optimization seconds | 4.45 | 117.64 |

Both arms fail exact code, radius and closure checks. Hard-code mismatch is over
all123,642,856 values reencoded with candidate radius; closure violation is over
120,653,800 positive-observed-radius values using fixed observed cells. These are
different metrics. Final label-class multiset overlap is11/16 in both arms,
which is not the matched per-image accuracy. The paired soft/hard objectives
are evaluated on the same final image and use the same public floor, before
applying lambdaQ. The hybrid's relaxed-objective improvement did not transfer
to hard-label consistency or image quality in this short fixed-schedule pilot.
It does not prove that the method is impossible or that label relaxation is its
only limitation. No additional experiment was automatically started.
