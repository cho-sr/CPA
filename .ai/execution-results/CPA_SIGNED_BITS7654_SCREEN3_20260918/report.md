# CPA_SIGNED_BITS7654_SCREEN3_20260918

Status: **FAILED**.

Execution is strictly serial in the registered order 7 -> 6 -> 5 -> 4.
Each bit uses a fresh observation from the same archived raw update; no candidate, target, or optimizer state crosses bit widths.

| Bit | Status | Evaluation |
| ---: | --- | --- |
| 7 | BLOCKED_BY_PREVIOUS_FAILURE | n/a |
| 6 | BLOCKED_BY_PREVIOUS_FAILURE | n/a |
| 5 | BLOCKED_BY_PREVIOUS_FAILURE | n/a |
| 4 | BLOCKED_BY_PREVIOUS_FAILURE | n/a |

Each public top3 was selected before private loading. Private evaluation consumes x only.
Actual-image cell feasibility: NOT_EVALUATED.
A failed bit stops all dependent lower-bit jobs; successful artifacts are preserved.
