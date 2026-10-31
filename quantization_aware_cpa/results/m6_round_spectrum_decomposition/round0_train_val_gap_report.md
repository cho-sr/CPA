run_tag="m6_round_spectrum_decomposition"

# Round 0 Train/Val Gap Audit

Round 0 train loss is `0.7276` and val/measurement loss is `1.1856`.

Findings from logs/code:
- `training_loss_mean` is not a moving average. The code calls `mean_probe_loss` at each checkpoint.
- Round 0 contains no optimizer step before measurement. The round loop measures checkpoint 0 before `train_one_round` is called.
- Train and measurement splits are disjoint (`train` vs `val`) and `measurement_batches_are_used_for_training=False`.
- The cached train and val batches use the same class-label sequence, so class distribution does not explain the gap.
- The image paths differ (`train/...` vs `val/...`), so the remaining explanation visible in logs is split/image difficulty.
- Per-train-batch checkpoint losses were not logged; only the five-batch train mean is available without recomputation.

Val batch losses at round 0: 0.8374, 1.4196, 1.1387, 1.4472, 1.0852.

Cached label audit:
- train total/unique labels: 160/160
- val total/unique labels: 160/160
- train/val label overlap: 160
- same labels in each matched batch: [True, True, True, True, True]
- same image paths in each matched batch: [False, False, False, False, False]
