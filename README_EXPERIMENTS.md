# Corrected Quantized-Update CPA Experiments

This directory reconstructs experiments 01--05 under the corrected definitions.
The immutable public author snapshot is the repository-root `src/` directory.
Files under `src/` are not modified by these experiments.

## Threat Model And Oracle Policy

Attack code may use only the observed update, public model/checkpoint, public
collector settings, and known labels only when an experiment explicitly declares
a known-label threat model. True embeddings, images, labels, and oracle
permutation matching are allowed only in final evaluation files.

Embedding CS is reported as:

`Oracle permutation-aligned cosine similarity, evaluation only`

The author `src/attack.py` top-level function is not used for 01--03 because it
reorders recovered sources with oracle evaluation before FIA. Instead,
`quantized_update_cpa/author_runner.py` calls the original author
`CocktailPartyAttack` and `Direct` classes directly and keeps evaluation outside
the attack loop.

## Experiments

01 `01_fp32_update_cpa.py`

Unmodified author CPA attack on our FP32 FedAvg updates. This is not a full
reproduction of the author's update collector. It uses the shared external
FedAvg collector and records that CPA factorizes a multi-step FedAvg update only
as an empirical baseline approximation. The public author negentropy behavior is
preserved: `src/gradient_inversion.py` computes `loss_ne`, then overwrites it
with zero.

02 `02_naive_4bit_update_cpa.py`

Derives int4 from the exact same FP32 artifact used by 01:

`s = max(abs(delta_w)) / 7`, `q = clip(round(delta_w / s), -7, 7)`,
`dequant = q * s`

Scales are per parameter tensor, zero point is absent, codes are `-7..7`
(15-level symmetric int4), and only nearest rounding is supported.

03 `03_randomized_dequant_ensemble_cpa.py`

Creates heuristic randomized dequantization members from the stored int4 codes.
Interior codes use `q*s + U(-s/2, s/2)`. Saturated codes `q=+7` and `q=-7` are
kept at `q*s`; this is explicitly not exact posterior sampling. Ensemble
alignment is blind: recovered sources are aligned by absolute cosine + Hungarian
assignment against the first or medoid member, then sign-aligned and aggregated.

04 `04_quant_aware_ica_objective_cpa.py`

Separate quantization-aware ICA proposal. It does not use a free `A,S` joint
factorization. It optimizes an unmixing matrix on a whitened attack-layer
observation and, when enabled, lets the observation candidate move inside the
observed quantization bins. Quantization consistency is squared distance to the
nearest-rounding bin:

Interior: `[(q-0.5)s, (q+0.5)s]`

Saturation: `q=7 -> [6.5s, +inf)`, `q=-7 -> (-inf, -6.5s]`

Negentropy uses standardized sources per component:

`s_tilde = (s - mean(s)) / (std(s) + eps)`

`J = (E log cosh(s_tilde) - E log cosh(v))^2`, `v ~ N(0, 1)`

Four ablations are supported: Q off/on crossed with NE off/on. Optional image
reconstruction after 04 is a separate Direct FIA stage, not a fully joint
end-to-end method.

05 `05_quant_aware_fedavg_refinement_cpa.py`

CPA-initialized quantization-aware FedAvg update-matching refinement. Known-label
mode fixes the true labels. Unknown-label mode uses trainable logits with
`softmax`; raw real-valued label matrices are never passed to cross entropy.

The update loss differentiably unrolls the same full-batch local SGD used by the
collector:

`theta_{t+1} = theta_t - eta * grad L(theta_t; X_hat, Y_hat)`

`delta_hat = theta_T - theta_0`

`delta_hat` is compared to all observed int4 bins for all model parameters.
If the update was produced from 128 samples, 05 optimizes all 128 samples for
update consistency. `--display_samples 16` only limits saved/displayed examples.

Unsupported conditions are hard errors: non-nearest rounding, momentum,
weight decay, non-full-batch local SGD, BatchNorm/stateful layers, or active
dropout.

## Outputs And Cache Safety

Canonical outputs go under `quantized_update_cpa/outputs_v2/`.
Every collector, derived update, and experiment run writes a `manifest.json`.
Manifest checks refuse incompatible reuse. `--fresh_start` deletes the relevant
attack/checkpoint/metric directory, including whitening cache and intermediate
reconstructions.

Manifest fields include experiment id, method, checkpoint path/hash, update
hash, sample index hash, label hash, model, attacked layer, local steps/epochs,
batch size, learning rate, optimizer, momentum, weight decay, bit width,
rounding, quantization-scale hash, seed, author source hash, and evaluated
sample counts.

## Commands

Small checks:

```powershell
.\run_sanity_checks.ps1
.\run_01_03_smoke.ps1
.\run_04_smoke.ps1
.\run_05_known_label_smoke.ps1
```

Full commands are printed by:

```powershell
.\run_full_experiments.ps1
```

That script does not launch full VGG16 runs automatically. VGG16 with
`n=128`, 10 local steps, and especially 05 `create_graph=True` can require very
large GPU memory.

## Legacy Files

The old QGM and randomized FIA postprocessor entrypoints now exit with
deprecation messages. Existing historical result directories are not deleted,
but they should not be mixed with corrected 01--05 results because their
manifests and oracle/cache policies differ.
