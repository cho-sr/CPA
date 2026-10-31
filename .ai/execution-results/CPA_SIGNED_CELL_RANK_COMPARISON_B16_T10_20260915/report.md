# CPA_SIGNED_CELL_RANK_COMPARISON_B16_T10_20260915

## Outcome

[FACT] This task used the user-designated existing Luna/xhigh Executor thread `01a09e4d-51f3-7121-84e2-03a7a9ac3039` (PID 18272). The pre-claim handoff snapshot matched exactly: `d4f3e424a4ad290351ef8409bd0cd8592a001369c62d5a7c6f9d8cf866894b8a`. The handoff was READY/unclaimed, then claimed by this thread and marked RUNNING before execution.

[FACT] The one-shot execution wrapper ran the authorized sequence serially. AP5 preprocessing and the original CPA 25,000/FIA 25,000 attack completed. The final offline evaluator failed before loading private true-G or evaluating images because `signed_cell_rank_evaluation.py:297` raised:

```text
NameError: name 'legacy_naive' is not defined
```

The wrapper stopped immediately with process exit code 1. No arm or stage was restarted, and no scientific condition was changed. The task result is `INCOMPLETE_STOPPED_ON_FAILURE`; `FULL_EXPERIMENT_COMPLETED` is not claimed.

## Stage evidence

| Stage | Status | Duration | Exit | Evidence |
|---|---|---:|---:|---|
| AP5 external uncentered rank-16/cell preprocessing | COMPLETED | 47.188 s | 0 | `preprocess.stdout.log`, `preprocess.stderr.log`, `quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5/` |
| Original CPA 25,000 then Direct FIA 25,000 | COMPLETED | 2,175.984 s | 0 | `attack.stdout.log`, `attack.stderr.log`, `quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5/` |
| Final offline comparison against stored naive | FAILED before private load | 2.485 s | 1 | `evaluation.stdout.log`, `evaluation.stderr.log` |

The authoritative wrapper record is `stage_execution.json`; the callback evidence is `.ai/relay-runs/completions/CPA_SIGNED_CELL_RANK_COMPARISON_B16_T10_20260915/comparison/status.json` and records process exit 1 with zero model calls while waiting.

## Completed preprocessing and attack artifacts

[FACT] The finalized AP5 public artifact passed the repaired V2 guards and records rank 16, AP5, `centered=false`, `rank_backend=gram_eigh`, FP32 CUDA projection, final-only selection, and metric-only scale. Public manifest SHA256 is `9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69`; checkpoint SHA256 is `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`; final V file SHA256 is `7b4850a13dc725017eab7da451019b1a2c3e90bd6d58d8d4a0db3181fcd010cf`; final V tensor SHA256 is `1b6790c2a17319c0f4069ed9f4325e5536ccbed02230d9abcc60536fe9181326`.

[FACT] The completed cell-rank result is `result.json` under `cell_rank_ap5`, with candidate SHA256 `700beaa7b7da8565da68bb782ea67738542695334d09a2910e29f83bf29ab9d6`, `cpa_status=COMPLETED`, `fia_status=COMPLETED`, `cpa_iterations=25000`, `fia_iterations=25000`, `image_count=16`, `labels_used=false`, and `private_inputs_used=false`. The stored naive arm remains in `quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive` and was reused only for the intended final evaluator; it was not rerun or modified.

[FACT] AP5 cell diagnostics remained finite and cell-feasible. At the final fifth step, `exact_code_ratio=1.0`, `bounded_cell_satisfied=true`, `bounded_cell_violation_count=0`, movement from V0 relative L2 was `0.3459331105004968`, movement from the preceding V was `0.02013763907749731`, `V_next-M_t` L2 residual was `0.002805809723213315`, and candidate scale was `5.0294027460040525e-05` versus observed `5.416279236669652e-05` (relative scale drift `0.07142846108198095`, metric only). These are diagnostics, not selection or success gates.

[FACT] Attack runtime instrumentation reports NVIDIA GeForce RTX 5060 Ti, CUDA 13.2, torch `2.13.0+cu132`, peak allocated CUDA memory `3,593,577,984` bytes and peak reserved CUDA memory `3,841,982,464` bytes. Internal CPA wall time was about 42.648 s and FIA wall time about 2,124.175 s; the wrapper stage duration includes setup and artifact writes.

## Offline comparison status

[FACT] Because the evaluator failed before `_load_true_update` and before image/embedding evaluation, the following requested results are unavailable and must remain uncomputed:

```json
{
  "embedding_cosine": {"status": "NOT_COMPUTED", "value": null},
  "lpips": {"status": "NOT_COMPUTED", "value": null},
  "psnr": {"status": "NOT_COMPUTED", "value": null},
  "ssim": {"status": "NOT_COMPUTED", "value": null},
  "true_g_relative_error": {"status": "NOT_COMPUTED", "value": null}
}
```

No true-G, image, embedding, private label, or reconstruction input was loaded by the failed evaluator. Consequently, no naive-versus-cell metric deltas or scientific improvement claim can be made. The intended zero-reference convention remains documented in the repaired evaluator, but it was not reached in this run.

## Failure detail and limits

[FACT] The traceback is in `evaluation.stderr.log`: `evaluate()` calls `validate_provenance()`, `_check_arm()` reaches line 297, and references the undefined name `legacy_naive`. This is a code failure in the final evaluation path, not a data, finite-value, cell, CUDA, CPA, or FIA failure.

[INFERENCE] The AP5 preprocessing and CPA/FIA artifacts are usable for a future explicitly authorized evaluation-only continuation after the evaluator defect is reviewed. They are retained in place; this task does not repair or rerun them.

[PROPOSAL] Review the evaluator's `_check_arm` compatibility variable at line 297, then issue a new bounded handoff for an evaluation-only continuation if desired. Any continuation should validate the existing finalized artifacts before loading private true-G and should not rerun the completed AP5/CPA/FIA stages.

[UNKNOWN] Embedding cosine, LPIPS, PSNR, SSIM, and true-G relative error remain unknown because the final offline stage did not execute. No claim about recovery quality is supported by this failed comparison.

## Integrity and protected sources

[FACT] Post-execution protected hash verification passed with `protected_match=true`; the retained `signed_cell_rank.py` SHA256 is `7ae6deb49738914f4cd1f9328397400d89835ede4cd6a462b2ea260a77ae4acb`. Evidence is `protected_hash_check_final.stdout.log` and `protected_hash_check_final.exit_code`. The original CPA/whitening/loss/FIA/codec/baseline sources, fixed conditions, old results, `.ai/STATE.md`, and `.ai/RELAY.md` were not modified.

[FACT] Execution-only files for this task are the serial wrapper `run_comparison.py`, pre-claim/configuration manifest `run_manifest.json`, `stage_execution.json`, per-stage stdout/stderr logs, this report, and `results.json`. Existing AP5 and cell-rank output directories are preserved; the requested evaluation output directory was never created because evaluation stopped before writing it.

## Planner delivery

[FACT] The bound Planner was rechecked as the existing `codex.exe` thread `01a09ac2-0e8d-7f90-94a8-c45be99f4343` (PID 12728) with its pinned process identity. Exactly one user-authorized final failure notification was accepted by the installed queue: `Queued message 01a0a196-ea4a-7cd0-ac5c-415086d3fdac`. No retry, rebind, monitoring loop, or additional work was performed.
