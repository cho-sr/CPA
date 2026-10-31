# FedAvg naive CPA + FIA: direct upstream baseline

User authorization: "FedAvg naive CPA 짜보자". This new baseline supersedes the
earlier restriction that reserved the word `naive` for no-CPA controls. Here,
`FedAvg naive CPA` explicitly includes CPA. Existing no-CPA results remain
named `no-CPA`; historical method keys and numeric results are preserved.

Run tag: `m8_fedavg_naive_cpa_dither4_B16`.

## Fixed experiment

- Same VGG16 IMAGENET1K_V1 weights, frozen convolutional representation.
- n=16, T=10, five independent saved actual client updates, all80 originals.
- Read the exact `quantized.pt/Y_hat` used in `m7_uia_4bit_B16`. The client
  update is `(W_initial-W_final)/lr`; no ideal factorization is substituted.
- The observation is the existing per-row 4-bit-interval dither model
  `Y + Uniform(-delta/2,delta/2)`, `delta=2*row_max_abs(Y)/15`. It is **not an
  integer 4-bit stochastic-rounding codec**. This baseline holds the observation
  fixed to examine the attack implementation. It does not introduce a new codec.
- **No LS-A calculation at all**, no debias, no MP threshold/truncation/diagnostic,
  no Stage2, no refinement, no source replacement, no new client training.

## Attack implementation and evaluation boundary

The adapter loads the actual local `official_cpa/src/gradient_inversion.py`
`CocktailPartyAttack` class and calls its constructor, `step`, and `get_rec`.
Its original whitening uses fp32 covariance divided by d-1, `torch.linalg.eig`,
and the top16 absolute eigenvalues. There is no shared qacpa Stage1 call.
W initializes as identity; CPA runs25000 Adam steps, lr0.001, no scheduler,
decor5.3/T7.7/nv0.13/l1=5/tv0.1/ne1. The original code itself overwrites the
neg-entropy term to zero and disables source TV when nv>0; preserve that behavior.

The signed recovered S is saved. `abs(S)` in estimated source order is passed
to the actual upstream `feature_inversion.Direct` class. FIA uses25000 steps,
TV0.1/FI1/Adam lr0.1/cosine scheduler, seed42 per batch. VGG16 features/avgpool
are identical to the upstream feature map; the unused classifier forward is
omitted. Only image pixels are optimized. The backend APIs accept neither
true images nor true embeddings.

Only **after FIA completes**, the evaluator loads the true images/embeddings.
Full16 Hungarian absolute-cosine matching establishes a permutation for metrics
and figures. No true-source scale or source substitution is applied. Both
estimated-source-order reconstructions and evaluation-aligned reconstructions
are saved, and all80 images are scored.

This correspondence adapter is explicitly different from the upstream script's
greedy matching before FIA. It also differs from old GIA's truth-derived source
scaling. Moving the permutation changes which source receives which initial
random image. Thus this is a direct execution of the **original CPA and FIA
algorithms with an observed-update/evaluation adapter**, not a byte-identical
execution of the original end-to-end CLI.

Earlier M7 uses the port's fp64 covariance/d/eigh whitening and Hungarian
permutation before FIA. Comparison with M7 changes the backend and correspondence
timing together. It must **not** be interpreted as an isolated LS-A ablation.
The old LS-A value did not feed the old FIA target either.

## Artifacts and execution

Code: `quantization_aware_cpa/qacpa/upstream_naive.py` and
`quantization_aware_cpa/qacpa/run_m8_fedavg_naive.py`.
Config: `quantization_aware_cpa/configs/m8_fedavg_naive_cpa_dither4_B16.json`.

From `quantization_aware_cpa` with the established DLG Python environment:

```powershell
python -m qacpa.run_m8_fedavg_naive --phase prepare
python -m qacpa.run_m8_fedavg_naive --phase run
```

`scripts/launch_m8_fedavg_naive.ps1` starts a hidden worker. Integer child exit
codes plus complete80-image output/hash checks determine completion. Upstream
optimizer checkpoints permit resumption with unchanged input/config/source
fingerprints. New manifests record the frozen input and upstream source hashes.

Outputs: per-image and per-batch CSVs; S tensors; source-order and aligned
reconstruction tensors; original CPA/FIA logs; comparison with the same80 M7
images; landscape all80 PNG and fixed-seed random12 PNG. LPIPS, PSNR, SSIM and
embedding |cos|>0.9 counts are reported. No MP-based B_eff prediction is computed.

The optional torchtext import is disabled only while loading the original
upstream datasets module, using its existing ImportError fallback. This avoids
an unrelated incompatible text-library DLL and does not alter ImageNet code.
