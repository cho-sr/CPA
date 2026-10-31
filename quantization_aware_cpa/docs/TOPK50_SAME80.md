# Top-k 50% sparsification with unchanged CPA + FIA

The user authorized this condition on 2026-09-11: replace the completed same80
packed 4-bit compression with magnitude Top-k 50% sparsification, retaining the
same client updates and attack. Run tag: `cpa_fia_topk50_same80_B16`.

The original 80 images remain in original order, in five independent n=16
client updates from round 0. Each cached update used VGG16 IMAGENET1K_V1,
frozen convolutional features, all classifier parameters updated by SGD
lr=0.001 for 10 steps, momentum/weight decay/dropout disabled. No retraining
or regrouping occurs. The observed tensor is the actual first-FC update
`Y=(W_initial-W_after10)/0.001`, shape 4096 x 25088. There is no server
averaging of the five measurements before the attack.

Compression is deterministic global magnitude Top-k **within that FC1 tensor**,
not per row or across the full model. Exactly 51,380,224 of 102,760,448
coordinates are retained. Cutoff ties prefer lower row-major flat indices.
Selected values retain their original FP32 bits; dropped coordinates decode
to zero. No quantization, rescaling, error feedback, or desparsification prior.
Selected slots and actual nonzero counts are recorded separately, since a
zero-valued coordinate can be selected when ties reach zero.

The real binary packet uses a 32-byte little-endian header (`<8sIIQQ`, magic
`QACTOPK1`, rows, columns, retained count, zero reserved field), a row-major
little-bit-order bitmap, and little-endian FP32 values in ascending position
order. The receiver takes only these packet bytes. Each packet has
205,520,896 value bytes + 12,845,056 bitmap bytes + 32 header bytes =
**218,365,984 bytes**. The corresponding dense FP32 packet has 411,041,824
bytes: about 1.88235x compression / 46.875% fewer bytes. This is packet
accounting for FC1 only, not full-model traffic or network latency.

CPA and FIA are the identical upstream calls used by the same80 packed4 run:
25,000 iterations each, unchanged coefficients/optimizers/schedulers,
CPA seed `0+batch_index`, FIA seed 42. All 16 estimated sources feed FIA in
estimated order through `abs(S_hat)`. Only after FIA, the unchanged signed
source absolute-cosine Hungarian assignment pairs all 16 images for metrics.
No LS-A, debiasing, MP truncation/diagnostic, Stage2, oracle fills, or extra
attack arms. Code/config/source hashes are frozen before attacks; the runner
also checks the completed packed4 implementation fingerprints.

Run or resume from **CMD**:

```bat
cd /d "C:\Users\조상제\Desktop\CPA\quantization_aware_cpa"
scripts\run_topk50_same80.bat
```

Equivalent direct command:

```bat
"C:\miniconda\envs\DLG\python.exe" -u -m qacpa.run_topk50_same80 --phase run
```

`--phase prepare` freezes/verifies inputs; `--phase packet` creates and audits
all five real packets on CPU; `--phase run` performs the requested five CUDA
attacks and then automatically reports all80; `--phase report` regenerates
the full report from completed artifacts. Only fully completed batches are
reused on resume; an interrupted batch re-enters the same seeded attack and
the existing upstream implementation resumes compatible stage checkpoints.

Output directory: `results/cpa_fia_topk50_same80_B16`. It contains execution
status, per-batch packets/diagnostics/float reconstructions/metrics/logs,
and, on completion, full80 and per-batch reconstruction PNGs, per-image and
per-batch LPIPS/SSIM/PSNR/source cosines, communication accounting,
distributions, `summary.json`, and one appended `results/runs.jsonl` row.

This condition is paired to the same80 packed 4-bit attack. Historical dense
32-bit results use a different CPA implementation/FIA matching order, so
that historical comparison does not isolate compression alone.

The actual pre-attack packet audit found nonzero fractions of 38.2171%,
39.1749%, 41.5865%, 42.0948% and 38.7438% in batches0-4. Thus Top-k 50%
drops only zeros in all five updates: decoded relative L2 error is 0 and
retained energy is 100%. This is lossless sparse representation for these
observations. The requested condition remains unchanged. The full audit is
saved as `sparsification_input_audit.json` and `.md` in the output directory.
