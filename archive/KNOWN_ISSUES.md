# Known issues preserved from the source workspace

Static inspection on 2026-10-31 found the following pre-existing issues. These
files are archived unchanged. No scientific code repair or experiment rerun was
performed as part of publication. Exact identities are in `KNOWN_ISSUES.json`.

| File | Line | Existing issue |
| --- | --- | --- |
| `quantization_aware_cpa/qacpa/run_pc_vs_pt_small_batch.py` | 43 | The unfinished script ends with a bare `def` statement and cannot run as Python. |
| `.ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916/final_audit.json` | 598 | A literal backslash-n follows the closing JSON brace. |
| `.ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917/before/parent_and_comparison/parent_final_audit.json` | 598 | The historical copy contains the same trailing suffix. |
| `quantization_aware_cpa/results/paper_sparse_same80_keep50_30_20_R20_20260921/execution_record.json` | 77 | A literal backslash-n follows the closing JSON brace. |
| `quantization_aware_cpa/results/paper_sparse_same80_keep50_30_20_R20_20260921/stage_execution.json` | 77 | A literal backslash-n follows the closing JSON brace. |
| `quantization_aware_cpa/results/paper_sparse_same80_keep50_30_20_R20_20260921/v1_supersession.json` | 69 | A literal backslash-n follows the closing JSON brace. |

Treat these as historical artifacts requiring review before use. The manuscript
dependency check and archived-byte hash checks are separate from these issues.
