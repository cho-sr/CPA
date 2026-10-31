# CPA signed cell-rank implementation repair report

`TASK_ID=CPA_SIGNED_CELL_RANK_REPAIR_20260915`, plan version 1. The existing
thread `01a09e4d-51f3-7121-84e2-03a7a9ac3039` was verified as the Luna/xhigh
Executor (PID 18272). Before editing, the handoff was `READY`/unclaimed and
its snapshot SHA256 was
`da1b04b625b213c6a4893bcb1f8ebf39e3c95ee26c193b4649ce3ddd9b52d569`; it was
then claimed by that exact thread and marked `RUNNING`.

## Scope and outcome

[FACT] Only the six handoff-authorized implementation/test/documentation files
were edited, plus this task's result directory. The parent plan, fixed
conditions, original CPA whitening/loss/optimizer, Direct FIA, codec/cell
utilities, baseline artifacts, and protected source files were not changed.

[FACT] R1--R5 are repaired. The task result is `COMPLETED`, while the research
experiment result remains `FULL_EXPERIMENT_NOT_RUN`.

## Repairs

- **R1 canonical input hash:**
  `quantization_aware_cpa/qacpa/signed_cell_rank_runner.py:299-305` now
  registers `upstream_naive._tensor_hash(V)`, the same hash used by the
  original CPA manifest. Preprocessing tensor/file hashes remain separate.
  The mock integration writes and checks the real upstream manifest path.
- **R2 figures and numeric preservation:**
  `signed_cell_rank_evaluation.py:337-357` uses a `(4,12)` compact grid and
  preserves `(16,3)` full layout. `evaluate()` commits `comparison.json` and
  `report.md` atomically before plotting, verifies both nonempty figure files,
  and records `evaluation_status=FAILED`/`plot_status=FAILED` with retained
  numeric results when plotting fails.
- **R3 movement logs:**
  `signed_cell_rank.py:281-309,365-385` retains `before` and computes
  `movement_from_previous_V = _movement(after, before)` for consecutive
  iterates; cumulative movement from `V0` is unchanged.
- **R4 stage statuses:**
  `signed_cell_rank_runner.py:98-131,259-280,307-399` records preprocessing,
  attack, evaluation, CPA, and FIA states. A completed attack awaiting the
  separate evaluator is `attack_status=COMPLETED`, `evaluation_status=NOT_RUN`;
  a failed FIA retains `cpa_status=COMPLETED`, `fia_status=FAILED`, and an
  overall failed attack result. The evaluator rejects incomplete new arms.
- **R5 integration/configuration guards:**
  `signed_cell_rank_evaluation.py:39-75,135-184` requires the exact registered
  and executed CPA/FIA settings, including learning rates, loss/configuration
  fields, seed, source hashes, canonical input fingerprint, cache containment,
  initialization, and runtime checks. Mock integration covers successful
  provenance/evaluation/figure output and negative tampering paths.

## Defect reproduction evidence

[FACT] `defect_reproduction.stdout.log` retains tiny probes for each reviewed
defect. The old compact `(4,9)` indexing raises `IndexError` at column 11; the
old core dtype hash and upstream canonical hash differ for the same FP32 tensor;
and a lagging two-step movement produces relative L2 `2.0` versus the repaired
consecutive value `0.5`. The status probe records the intended completed,
pending-evaluation and failed-FIA states.

[INFERENCE] These probes reproduce the reviewed pre-fix expressions and make
the failure modes explicit; they do not execute a real arm or claim scientific
performance.

## Validation

All commands used `C:/miniconda/envs/DLG/python.exe`, synthetic CPU tensors,
temporary artifacts, mocked CPA/FIA, and mocked LPIPS/image metrics only.

- `pytest quantization_aware_cpa/tests/test_signed_cell_rank.py quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py -q --basetemp .tmp_pytest_signed_cell_rank_repair`: **14 passed** in 7.11 s. This includes all16 Agg rendering, injected savefig failure with retained numeric JSON, canonical fingerprint and learning-rate tampering rejection, failed-stage persistence, movement checks, and the successful mocked end-to-end provenance path. Output: `tests_new.stdout.log`.
- `pytest quantized_update_cpa/tests/test_cell_utils.py quantization_aware_cpa/tests/test_upstream_naive.py quantization_aware_cpa/tests/test_signed_baseline_provenance.py -q --basetemp .tmp_pytest_repair_existing`: **109 passed** in 5.99 s. Output: `tests_existing.stdout.log`.
- The five edited Python modules/tests parse successfully; output: `syntax.stdout.log`.
- The existing naive baseline's CPA/FIA result and manifest settings equal the
  exact guard dictionaries (CPA25k/lr 0.001; FIA25k/lr 0.1); output:
  `baseline_settings_check.stdout.log`.

[FACT] The protected-source comparison in `protected_hashes.json` is an exact
match (`match=true`) to the parent report's `protected_source_hashes_current`.

## Artifact and stopping status

[FACT] Retained evidence is under
`.ai/execution-results/CPA_SIGNED_CELL_RANK_REPAIR_20260915/`, including test
stdout, exit codes, defect probes, syntax output, and protected-hash output.
No full B16/T10 preprocessing, CPA/FIA, GPU run, real observation/private
evaluation, download, worker, or new AI session was started.

[UNKNOWN] Full-size resource use, CUDA behavior, real movement/convergence,
CPA/FIA impact, LPIPS on real images, and image-quality results remain unknown
until a separately authorized research run.

[PROPOSAL] A future explicit research handoff may use the documented AP5/CUDA,
original CPA25k/FIA25k, and label-free offline commands; they were not run here.

[FACT] The visible Planner binding in `.ai/relay-runs/visible/peers.json` is
the stale historical thread `01a09ac2-0e8d-7f90-94a8-c45be99f4343` (PID
12728). Per the handoff's stale-binding rule, no queue retry or replacement was
attempted; final delivery is recorded as `FINAL_RESULT_HANDOFF_UNCONFIRMED`.
