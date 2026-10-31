# CPA_SIGNED_CELL_EVALUATION_REPAIR_20260915

## Scope and completion

[FACT] This continuation used the existing user-designated Luna/xhigh Executor thread `01a09e4d-51f3-7121-84e2-03a7a9ac3039` (PID 18272). The pre-claim handoff SHA256 matched `7dbb5e5da4b6b1b81b021e5dbcb7ca088641aa3876d5d81981367b391e27191c`; the READY/unclaimed handoff was claimed by that exact thread and marked RUNNING.

[FACT] The authorized minimal repair changed `_check_arm` in `quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py:297` from the undefined `legacy_naive` name to the existing `naive` parameter. The compatibility contract is unchanged: a legacy naive result may omit its duplicated result-level checkpoint field and relies on its validated registration hash; new cell results must provide and validate the result-level field. No provenance guard was weakened.

[FACT] Focused regression tests and syntax/import checks passed. The saved AP5 preprocessing and completed CPA25,000/FIA25,000 artifacts were reused without rerun. The final offline evaluator then completed successfully, including provenance-gated private true-G, image, embedding, LPIPS, PSNR and SSIM evaluation. The handoff is marked COMPLETED with `FULL_EXPERIMENT_COMPLETED` for the parent comparison continuation.

[INFERENCE] The comparison below is valid for this one batch/seed and the recorded execution. It does not establish generalization or a universal recovery improvement.

## Regression evidence

[FACT] The pre-fix parent evaluation log, retained as `pre_fix_traceback.log`, ends at line 297 with `NameError: name 'legacy_naive' is not defined`. After the one-line repair:

* `test_legacy_naive_without_result_checkpoint_reaches_offline_evaluation` exercised the actual legacy branch and reached completed offline evaluation.
* `test_cell_arm_without_result_checkpoint_rejected_before_private_load` verified that a non-legacy cell arm missing the result-level checkpoint is rejected.
* `test_checkpoint_tampering_rejected_before_private_load` covered malformed/wrong registered and result checkpoint values for both arms with a private-load sentinel.

The final focused command was run with the required package paths in the process environment:

```text
C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_signed_cell_rank.py quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py quantization_aware_cpa/tests/test_signed_cell_rank_validation.py -q --basetemp C:/Users/조상제/Desktop/CPA/.tmp_pytest_signed_cell_evaluation_repair_03
```

Result: **49 passed**, 0 failed, 240 matplotlib deprecation warnings, 9.61 s, exit 0 (`tests_focused_03.stdout.log`, `tests_focused_03.exit_code`). Two preliminary invocations from an unsuitable import path exited 2 during collection; no research test or artifact was changed by those attempts. Syntax compilation and import checks both passed with exit 0 (`syntax.*`, `import.*`).

## Saved-result offline evaluation

[FACT] The exact authorized evaluator command was run once against the existing naive, AP5 cell-rank, public observation, private observation and new evaluation output path. `evaluation_execution.json` records exit 0, duration 20.328 s and `full_evaluation_status=COMPLETED`. The output contains `comparison.json`, `report.md`, `all16_comparison.png` and `all16_compact.png`.

[FACT] Public/final-arm provenance and initialization/runtime checks passed before private loading. Raw true-G was bound to `raw_updates.pt` SHA256 `ef9aefc66ab9fcfd0b023fdfc064dbadc27567bd3ef3508bc14ebda9af28ba4c`; evaluation truth SHA256 is `724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2`; the shared initialization fingerprint is `6f6ec493923b7081284f0fc405f76f60b583cc41a5977393751f96bb21147e59`. `offline_private_loaded=true` only after those checks, `labels_consumed=false`, all 16 candidates were included, and the same pixel-MSE Hungarian mapping was reused across image metrics.

### Mean metrics

| Metric (mean over 16) | Stored naive | AP5 cell-rank | Cell-rank minus naive | Direction |
|---|---:|---:|---:|---|
| Embedding cosine (oracle, evaluation-only) | 0.3116051623 | 0.3447613957 | **+0.0331562333** | higher |
| MSE | 0.0781093468 | 0.0779298795 | **-0.0001794674** | lower |
| PSNR (dB) | 11.5173321456 | 11.5045254289 | **-0.0128067168** | higher |
| SSIM | 0.2848939897 | 0.2816569652 | **-0.0032370246** | higher |
| LPIPS | 0.8229246028 | 0.7941681929 | **-0.0287564099** | lower |

[FACT] The complete per-image scores and the 16 truth-paired deltas are retained in `quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5/comparison.json`. Both arms report zero perfect pixel matches. LPIPS used the recorded torchmetrics VGG implementation on CPU with downloads disabled.

[INFERENCE] Image metrics are mixed: cell-rank improves MSE and LPIPS slightly, while PSNR and SSIM decrease slightly. Embedding cosine increases. These are descriptive deltas with no threshold or selection effect.

### True-G update error

[FACT] Both estimates use the same attack layer `classifier.0.weight` and delta convention `theta_after10_minus_theta_start`; `selection_used=false` for both.

| Metric | Stored naive `sq` | AP5 cell-rank `V` | Cell-rank minus naive |
|---|---:|---:|---:|
| Absolute Frobenius error | 0.0337123333 | 0.0291747845 | **-0.0045375488** |
| Reference Frobenius norm | 0.0460406595 | 0.0460406595 | 0 |
| Relative error | 0.7322295906 | 0.6336743394 | **-0.0985552511** |

The relative-error reduction is approximately 13.4596% of the naive relative error. The denominator was positive; the zero-reference JSON-null convention was not triggered in this run. True-G metrics remained evaluation-only and did not influence ranking, tuning, stopping or reconstruction.

## Preserved execution facts and diagnostics

[FACT] The parent AP5/cell-rank result remains unchanged: public manifest SHA256 `9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69`, checkpoint SHA256 `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`, final V file SHA256 `7b4850a13dc725017eab7da451019b1a2c3e90bd6d58d8d4a0db3181fcd010cf`, V tensor SHA256 `1b6790c2a17319c0f4069ed9f4325e5536ccbed02230d9abcc60536fe9181326`, and candidate SHA256 `700beaa7b7da8565da68bb782ea67738542695334d09a2910e29f83bf29ab9d6`.

[FACT] Existing attack diagnostics record AP5 rank 16, finite values, exact fixed-s code ratio 1.0, bounded-cell violations 0, movement from V0 relative L2 `0.3459331105`, final V-minus-M residual `0.0028058097`, and metric-only scale drift `0.0714284611`. The original attack runtime recorded peak CUDA allocation `3,593,577,984` bytes and reservation `3,841,982,464` bytes. No scientific condition or original CPA/whitening/loss/FIA/codec/baseline source was changed.

[FACT] Post-repair source/input hash evidence is in `pre_edit_hashes.json`: evaluator hash changed from pre-fix `db719026ee65b4e1c808da9f6abbaef9bb29d17da5d35f5e36aa816cbb36e7b8` to `c163aaa54bd8d5ebfc1528f49cafe2a830de72a9d7fe38583391b7ddaf4a65ef`; `signed_cell_rank.py` remains `7ae6deb49738914f4cd1f9328397400d89835ede4cd6a462b2ea260a77ae4acb`; saved attack/preprocess hashes remained unchanged. Protected original hashes were rechecked and matched.

## Files and limitations

[FACT] Scoped research changes are limited to:

* `quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py`
* `quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py`

Execution-only wrapper, hashes, logs, report and result files are under `.ai/execution-results/CPA_SIGNED_CELL_EVALUATION_REPAIR_20260915/`. Existing failed parent records and completed AP5/CPA/FIA outputs were preserved.

[UNKNOWN] This is one B16/T10 batch with seed 42. No claim is made about other batches, seeds, models or generalization. The embedding metric is explicitly an offline oracle metric; it was not used for candidate selection.

[PROPOSAL] Any later replication should use a new explicit handoff and preserve this completed output as the baseline; do not rerun the completed arms solely to repeat evaluation.

## Final Planner delivery

[FACT] The existing Planner binding was reverified as live `codex.exe` thread `01a09ac2-0e8d-7f90-94a8-c45be99f4343`, PID 12728. One authorized final completion/result notification was accepted by the installed queue: `Queued message 01a0a364-4c2a-7d53-bfd9-f72b71785712`. No monitoring loop, retry, rebind or additional experiment was performed.
