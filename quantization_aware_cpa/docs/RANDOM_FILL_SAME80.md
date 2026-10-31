# Authorized random-fill Top-k10 experiment (2026-09-12)

User selected **10% only**, independent uniform U(-tau,tau),20 random
candidates per original victim batch. Each candidate runs CPA25,000, then
FIA5,000; the top3 continue to total FIA25,000. The five original n16 batches
cover the same80 ImageNet images. There are100 CPA/screen candidates and15
final candidates, not100 victim batches. No new client data/update collection.

## Frozen victim and communication

Reuse the byte-identical existing packets at
`results/cpa_fia_topk10_same80_B16/batch00..04/observation.bin`.
They encode the actual Y=(W_initial-W_after10)/0.001 for classifier.0.weight:
VGG16 IMAGENET1K_V1 round0, conv frozen, all classifier parameters updated,
full batch16, SGD lr0.001,10 steps, no momentum/weight decay/dropout. The
server-model increment convention is -0.001*decoded_Y.

Top-k is global within FC1: floor(102760448*10/100) retained FP32 values,
ties by ascending flat index, bitmap positions. Per packet53,949,264 bytes;
dense equivalent411,041,824 bytes;86.874994% savings. Same packet, zero
extra transmission for random filling, no full-model traffic claim.

## Randomization and iteration budget

tau=min(abs(retained_values)) from the packet; all retained FP32 bits remain
unchanged. The independent missing-coordinate draws use numpy PCG64,
seed=100000+100*batch_index+candidate_index, chunk length1048576. Filling is
performed once before CPA, not resampled each iteration. Candidate0..19
is a full16-image CPA/FIA solution, never a selected image subset.
FP32 rounding at the uniform endpoints is clipped to nextafter(tau,0) in
magnitude so discarded fills cannot win cutoff ties on recompression.

CPA original upstream25k Adam lr0.001, no scheduler; decor5.3,T7.7,ne1,
nv0.13,l1=5,tv0.1; seed=baseline seed0+batch. Abs signed sources in estimated
order enter original Direct FIA: Adam0.1,TV0.1,FI1,seed42. All candidates use
the same CPA/FIA seeds to isolate random filling. Neither upstream objective
nor metrics/matching is modified. No LS_A/debias/MP/Stage2/oracle fills.

The cosine schedule ALWAYS has horizon25k, including screening. At5k Adam,
cosine scheduler, pixels and RNG state are checkpointed; finalists execute
20k additional steps. Screening is not an independent5k cosine schedule.
Per batch:20*25k CPA=500k;20*5k+3*20k FIA=160k;23 ranking replays.
The existing zero-fill baseline has only one CPA25k/FIA25k attempt. The
requested comparison therefore changes search compute as well as filling.
Do not attribute any improvement solely to random imputation. Do not run
additional baseline arms without a separate user request.

## Selection without ground truth

For each full16-image reconstruction, normalize exactly as ImageNet inputs,
compute frozen VGG features and labels predicted by the public round0
classifier. Replay the same10-step all-classifier SGD on those features.
Compute the actual FC1 weight difference divided by0.001, apply the SAME
Top-k codec, and compute relative L2 of decoded recompressed predicted Y
against decoded received Y (zero-fill on both sparse packets).

Minimize this score; ties use candidate index ascending. Screen all20 before
advancing the best3. After all3 reach25k, select one entire candidate per
victim batch using the same score. Predicted labels are a public-model
proxy, not recovered true labels; consistency need not track image quality.
No original images/labels/features/dropped true values are used for filling
or ranking. Freeze screen/final selections before loading evaluation truth.

Post-selection evaluation uses the unchanged signed-source absolute-cosine
Hungarian matching, without feature scaling/substitution, followed by the
same LPIPS/SSIM/PSNR on all16/all80. All20 screen and3 final reconstructions,
seeds, fill hashes, scores/predicted labels and checkpoints remain saved.
No oracle best-of-three/per-image metric cherry-picking.

## Run/resume in CMD

From quantization_aware_cpa:

```bat
scripts\run_random_fill_topk10_same80.bat
```

The runner holds an exclusive execution lock. Run the same command to resume
after a stopped process; code/config/source fingerprints must match. Do not
start a second copy while one is running. Status is execution_status.json;
each candidate fia/progress.json records actual iterations. Final outputs:
per-image/per-batch metrics, all80 original/reconstruction16:9 PNG,
all80_rec.pt, baseline comparison CSV, summary/report and results/runs.jsonl.

Disk use is bounded by sequential candidates; dense filled FC1 matrices are
regenerated from packet+seed and are not persisted. Tests check retained
bits, seed reproducibility, cutoff limits, update replay, selection without
truth, and staged optimizer/cosine continuation versus continuous execution.
