# CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917

Status: **PARTIAL / FAILED DURING PUBLIC CONTINUATION**.

Registration, sender conversion, all nine ICA separations, public alignment, center FIA25000, and all eight member FIA5000 screens completed.

The frozen public ranking was member 01, member 07, member 05. Member 01 completed its 20000 continuation updates from the authenticated 5000-step raw checkpoint and reached a saved 25000-step raw checkpoint. The isolated writer then raised `StagedFIAError` because it incorrectly required a new prefix checkpoint for a continuation child; its candidate/result files were therefore not written.

Executed optimizer updates: 85,000 of the registered 125,000. Member 07/member 05 continuation, private gate/evaluation, metrics and figures were not run. No restart, new arm, tuning or private feedback was performed.

Sender consumption: archived true updates were consumed for INT8 conversion. Attack consumed public q/s and settings only. Private evaluation did not start. Actual-image cell feasibility: **NOT_EVALUATED**.

See `final_audit.json`, `execution_summary.json`, and the preserved `parent_failure_evidence/` directory for hashes and stage evidence.
