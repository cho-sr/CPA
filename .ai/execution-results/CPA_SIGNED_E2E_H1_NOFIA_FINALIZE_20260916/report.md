# CPA_SIGNED_E2E_H1_NOFIA_FINALIZE_20260916

Status: **COMPLETED**.

The saved C_H1_high_noFIA candidate was finalized after the parent wrapper's exit 1 at the endpoint schema gate. No optimizer step was run during finalization; the original 100-step arm and all parent artifacts remain byte-identical.

Public endpoint authentication passed for x0, B_H1_high, and C_H1_high_noFIA. The canonical H1 profile uses exact integer, boolean, string, and tensor-schema checks; bounded float differences are recorded in endpoint_profiles.json.

The private all16 evaluation completed after the public gate. It consumed only private image tensor x; labels_consumed=false and true_updates_consumed=false.

Private report: quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916/report.md.
Archived parent failed summary SHA256: 379d79565906075db58da9fa4695e8adda907eb2d88a9ec59872443b8ba224be.

Validation: 17 focused tests passed; finalizer and private evaluator exited 0; new optimizer steps: 0.

Artifacts: execution_summary.json, results.json, tests.log, source_change_record.json, source_diff.patch, and the parent failed receipt archive under before/.
