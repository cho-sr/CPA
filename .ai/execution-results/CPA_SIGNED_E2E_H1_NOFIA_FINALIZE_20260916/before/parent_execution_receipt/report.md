# CPA_SIGNED_E2E_H1_NOFIA_20260916

Status: **PARTIAL / NEEDS_REVIEW**.

The focused validation and the single authorized 100-step `C_H1_high_noFIA` optimization completed. The public runner then stopped during archived `B_H1_high` endpoint-profile authentication, before writing endpoint profiles and before the private evaluator was started. No rerun was performed.

| Stage | Result | Exit | Wall seconds |
| --- | --- | ---: | ---: |
| Focused and relevant tests | 88 passed, 240 warnings | 0 | 11.458534 |
| Public noFIA runner | new arm completed; endpoint authentication stopped | 1 | 157.801875 |
| Private evaluator | not started | n/a | n/a |

The preflight checks passed exactly: initial H1 cell loss was `22.991680145263672`, feature norm was `21.787342071533203`, and the full-size canonical/adjoint terminal deltas and feature paths matched. The new arm used the fixed H1 labels, `lambda_fia=0`, `lambda_tv=0.1`, and `lambda_cell=0.24202997768915402` for 100 steps from the saved naive x0. Its candidate SHA256 is `5a20b87bf69473bf7696e46d9054c0f3d04a4c02f98ffde6ee3969374bf4e525`; peak allocated CUDA memory was 3,987,526,656 bytes.

The runner failure was `ExperimentError: archived B_H1_high endpoint profile mismatch`. The archived control was not retrained. The failure occurred in the new profile validator before `endpoint_profiles.json` was written; therefore this run does not provide a public endpoint comparison or a scientific FIA=1 versus FIA=0 decision. The private evaluator was not invoked and no private labels, true updates, or private images were loaded.

The callback waited once with zero model calls and queued its completion message to the existing Executor. The wrapper exited with status 1. Existing protected sources, the prior public screen, and its archived control remain preserved. The runner stderr contains one nonfatal PyTorch scalar-conversion warning in addition to the recorded validator failure.

Artifacts:

- Partial public output: `quantization_aware_cpa/results/signed_e2e_h1_nofia_B16_T10_20260916/`
- Private output: not created because the public phase did not complete
- Execution logs and summary: this directory
