# CPA_SIGNED_BITS7654_RECOVERY_20260918

Status: **COMPLETED**.

This recovery preserved the original failed wrapper evidence and used the saved bit7 computation.
Bit7 used zero new optimizer steps; bits 6, 5 and 4 were processed serially with fresh per-bit observations.

| Bit | Status | Evaluation | Candidate sets |
| ---: | --- | --- | ---: |
| 7 | COMPLETED | COMPLETED | 13 |
| 6 | COMPLETED | COMPLETED | 13 |
| 5 | COMPLETED | COMPLETED | 13 |
| 4 | COMPLETED | COMPLETED | 13 |

Every private evaluator ran after its public gate and consumed x only; labels and true updates were not consumed by the attack.
Actual-image cell feasibility: NOT_EVALUATED.
Recovery additional accounted stage time: 32652.286680 seconds.
Bit7 original remaining allowance: 1578.474900 seconds; aggregate original remaining allowance: 39375.645889 seconds.


## Post-exit artifact audit

The callback reported wrapper exit code 0. The raw `execution_summary.json` retains `error_type=RecoveryError` with `error=bit 6 registration bytes changed` from a late registration transition. That evidence is preserved. A separate final audit verified all four saved comparison files as `COMPLETED/EVALUATED`, each with `PUBLIC_GATE_PASSED`, `private_loaded=false`, 13 authenticated candidates and the registered stored Hungarian correspondence. No optimizer or replay step was rerun.

Corrected baseline-inclusive contact sheets were completed for bits 5 and 4 under their existing evaluation directories using saved candidates and stored matching. The combined available figure manifest now lists bits 8, 7, 6, 5 and 4 with no pending bits. The actual-image cell remains `NOT_EVALUATED`.

Final audit: `.ai/execution-results/CPA_SIGNED_BITS7654_RECOVERY_20260918/final_audit.json`.
