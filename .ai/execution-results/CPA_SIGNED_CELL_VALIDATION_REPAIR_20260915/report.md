# CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915

## Scope and status

[FACT] This is task `CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915`, plan version 1, executed as the user-designated existing `CODEX_EXECUTOR` thread `01a09e4d-51f3-7121-84e2-03a7a9ac3039` (PID 18272) with the verified `gpt-5.6-luna / xhigh` runtime. The handoff snapshot was verified before editing: `87e761304331f456c79d7e80bd47c57ddd621dea901eb446672f8ece33141678`. The READY/unclaimed handoff was claimed by that exact thread and marked RUNNING.

[FACT] The bounded work is complete. The handoff is being marked COMPLETED after this report and `results.json` are written. `FULL_EXPERIMENT_NOT_RUN` is explicit: no B16/T10 collection, GPU comparison, real-observation preprocessing, private-tensor load, reconstruction, or new worker/session was run.

[INFERENCE] The three validation holes are closed for the implemented public provenance and offline metric paths; the evidence below establishes contract behavior on synthetic/mock fixtures and existing adapter/provenance regressions, not scientific improvement.

## Repairs

### V1: initialization evidence

[FACT] `quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py:103-166` now validates field presence and exact JSON/Python types before equality checks, requires lowercase 64-character SHA256 values, checks the fixed `[16, 3, 224, 224]` shape and seed 42, and checks the complete baseline runtime contract. The contract retains `deterministic_warn_only=true` together with the existing cuDNN, TF32, CUBLAS workspace, and thread settings. Both arm initializations are validated and compared before the offline evaluation proceeds.

[FACT] The actual existing naive `fia_initial.json` was checked by `baseline_runtime_check_02.py` and passed (`baseline_runtime_check_02.exit_code=0`): shape `[16,3,224,224]`, seed `42`, and `deterministic_warn_only=true` were observed. Extra runtime metadata is retained; only fixed contract fields are compared.

[FACT] `test_signed_cell_rank_validation.py:19-50` covers a valid pair, every required field missing independently on both sides, invalid SHA/null, bool-as-int seed, wrong runtime types, and fixed-runtime mismatches. The pre-fix probe records the former holes: `old_missing_runtime_equal=True` and `old_bool_int_equal=True`.

### V2: preprocessing and arm cross-checks

[FACT] `quantization_aware_cpa/qacpa/signed_cell_rank_runner.py:107-196,293-419` independently validates registration metadata, preprocessing result metadata, and the declared arm before the original CPA call. It requires rank 16, `centered=false`, `gram_eigh`, CPU/CUDA FP32 projection metadata, final-iteration-only selection, metric-only scale, a declared AP iteration of exactly AP0 or AP5, and matching public checkpoint/manifest, q/s shape and dtype, V file/tensor and diagnostics hashes, stage statuses, and registration/result fields. The CLI accepts `--expected-ap-iterations {0,5}` and defaults to AP5.

[FACT] `test_signed_cell_rank_pipeline.py:369-475` exercises AP mismatch before CPA, registration/result rank and metadata mismatches, AP5 acceptance, and invalid sender/cell V bindings. The pre-fix probe records `old_registration_result_partial_check=True` and no old AP0/AP5 declared contract. A private `torch.load` sentinel confirms invalid public bindings fail before true-G loading.

### V3: finalized offline true-G error

[FACT] `quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py:495-581,702-790` loads the private `raw_updates.pt` only after all public/final arm checks, binds its checkpoint, layer, hash, and `theta_after10_minus_theta_start` convention, and computes evaluation-only metrics for the finalized naive `sq` and cell-rank `V` inputs. The norm uses chunked CPU float64 accumulation. Metrics include absolute L2 error, reference norm, relative error, status, layer, convention, source hash, estimate hash, and `selection_used=false`; they do not affect candidate selection, tuning, stopping, or reconstruction.

[FACT] For a zero reference norm, the serialized schema is:

```json
{
  "absolute_l2_error": 0.0,
  "reference_l2_norm": 0.0,
  "relative_l2_error": null,
  "relative_l2_error_status": "UNDEFINED_ZERO_REFERENCE_NORM"
}
```

The same null/status convention is used when the estimate is nonzero; no epsilon, NaN, infinity, or finite ratio is invented. `test_signed_cell_rank_validation.py:52-86` covers positive norm, both-zero, nonzero-estimate/zero-reference, invalid dtype/shape/nonfinite values, and wrong checkpoint/layer/convention/raw shape. The pre-fix probe records `old_zero_reference=ZeroDivisionError` and the repaired null/status result.

## Validation evidence

All commands used `C:/miniconda/envs/DLG/python.exe` and were synchronous CPU unit/mock/regression checks. Only the final non-overlapping runs are counted.

| Check | Result | Evidence |
|---|---:|---|
| Focused signed-cell tests (`test_signed_cell_rank.py`, `test_signed_cell_rank_pipeline.py`, `test_signed_cell_rank_validation.py`) | **43 passed**, 0 failed, 144 matplotlib deprecation warnings, 6.75 s, exit 0 | `tests_focused_final.stdout.log`, `tests_focused_final.exit_code` |
| Existing codec/original-adapter/provenance regression (`test_cell_utils.py`, `test_upstream_naive.py`, `test_signed_baseline_provenance.py`) | **109 passed**, 0 failed, 1 upstream warning, 4.83 s, exit 0 | `tests_existing_final.stdout.log`, `tests_existing_final.exit_code` |
| Python syntax compilation | pass, exit 0 | `syntax_01.stdout.log`, `syntax_01.exit_code` |
| Existing baseline initialization/runtime check | pass, exit 0 | `baseline_runtime_check_02.stdout.log`, `baseline_runtime_check_02.exit_code` |
| Pre-fix hole reproduction (synthetic expressions only) | pass, exit 0 | `pre_fix_hole_reproduction.stdout.log`, `pre_fix_hole_reproduction.exit_code` |
| Protected source hash comparison | `protected_match=true`, exit 0 | `protected_hashes.json`, `protected_hash_check_final.stdout.log`, `protected_hash_check_final.exit_code` |

[FACT] The retained focused and regression logs are under `.ai/execution-results/CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915/`. Earlier failed/overlapping attempts remain as history but are not counted as final validation.

## Protected files and changed files

[FACT] The ten protected hashes from the parent implementation result match before/after exactly (`protected_match=true`):

```text
official_cpa/src/gradient_inversion.py        c598e1ee7b4079f741060d10c8dfd14711e393a97384c63cf32be77a458b1873
official_cpa/src/feature_inversion.py        dc73880427e3eaf1f873d295f17f1d48d301882f634ee5df36ddeea8e56c4dec
official_cpa/src/vgg16.py                    8114eca8cde553dace5f777aedd21b70851c570d473e597b005cb6be4b248d36
quantization_aware_cpa/qacpa/upstream_naive.py 8f29b885d86f4121eda5c55e17486a3a48bd8db08b2f83890fe7726730d1ee92
quantization_aware_cpa/qacpa/seed.py         769366fcc2cf3e144d8caa86f73a55dcb0b6be615a45dab8b6e117399150c558
quantization_aware_cpa/qacpa/signed_baseline.py a2a6b3e3c7909fb15ab44d49839e6bb0759e44832e223d16f0f2c24e0e45e467
quantization_aware_cpa/qacpa/signed_baseline_observation.py 2c18135e86e80c8f666033d5ea9d0c3495869ee60809fed04e295f28fa92cda2
quantized_update_cpa/quantization.py         ae0f812c7de3de07cce3796c701475d6fc3e0a822f7d6dd8ab951d34201e3ceb
quantized_update_cpa/cell_utils.py            6725bce108a8e795680342231696c70f1d2bc8b01a727022c605e40aa841d361
quantized_update_cpa/trial_metrics.py        37368d41d5b5caac142949493ab7f501e7d5aecf9ed5e317c7112cc4c4ca1e50
```

[FACT] The current `quantization_aware_cpa/qacpa/signed_cell_rank.py` hash is preserved and recorded as `7ae6deb49738914f4cd1f9328397400d89835ede4cd6a462b2ea260a77ae4acb`; its projection core was not changed.

[FACT] Scoped implementation/documentation/test files changed in this task:

* `quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py`
* `quantization_aware_cpa/qacpa/signed_cell_rank_runner.py`
* `quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py`
* `quantization_aware_cpa/tests/test_signed_cell_rank_validation.py`
* `quantization_aware_cpa/docs/signed_cell_rank_comparison.md`
* `.ai/execution-results/CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915/report.md`
* `.ai/execution-results/CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915/results.json`

Original CPA/whitening/loss/FIA/codec/baseline sources, fixed conditions, old results, `.ai/STATE.md`, and `.ai/RELAY.md` were preserved.

## Compatibility and limitations

[FACT] The public-only runner continues to feed the original CPA/FIA adapters and exposes explicit AP0/AP5 metadata. The finalized V path is referenced by registration/result file and tensor hashes and the canonical public input hash. The true-G path is evaluation-only and synthetic in this task.

[UNKNOWN] Real-size resource use, real private-observation availability, and scientific reconstruction/embedding improvement remain unverified because the user explicitly excluded those executions.

[PROPOSAL] A future authorized research invocation should use the documented AP5 command (or explicitly AP0), preserve the existing CPA25k/FIA25k configuration and candidate ordering, and report the new true-G fields without using them for selection.

[FACT] No principal-angle gate, loss, scale constraint, label/seed behavior, reconstruction policy, scientific condition, or extra arm was introduced.

## Planner notification

[FACT] The existing Planner binding was rechecked immediately before delivery: `.ai/relay-runs/visible/peers.json` CODEX thread `01a09ac2-0e8d-7f90-94a8-c45be99f4343`, PID 12728, is a live `codex.exe` process with the recorded creation time; recent visible deliveries succeeded. One authorized final result notification was accepted by the installed queue: `Queued message 01a0a161-5e3b-71c3-a6f5-43fd09b1bee0`. No retry, rebind, monitoring loop, or reply wait was performed.
