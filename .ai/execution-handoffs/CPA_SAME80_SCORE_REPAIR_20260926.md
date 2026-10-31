# Reviewed implementation repair and continuation of the claimed Same80 task

AMENDMENT_ID: CPA_SAME80_SCORE_REPAIR_20260926
TASK_ID: CPA_SAME80_BITS87654_R100_SCREEN3_20260926
PLAN_VERSION: 1 (scientific contract unchanged)
ASSIGNED_ROLE: CODEX_EXECUTOR
AUTHORIZATION: EXISTING_IMPLEMENT_TEST_EXECUTE_AND_AUTORESTART_SCOPE
STATUS: READY_FOR_EXISTING_EXECUTOR

The prior completion report is an implementation failure report, not completion
of the user's experiment. Automatic recovery correctly stopped on a deterministic
code error. Planner has now inspected that error and adjacent adapter defects.
Under THREE_CLI_ROLES.md, routine implementation fixes and necessary revalidation
within the already approved experiment remain authorized. No additional user
approval, scientific condition change, new claim or replacement session is needed.
Finish these narrow repairs and resume the remaining approved work through the
single installed supervisor. Do not merely repeat the NEEDS_REVIEW report.

## Verified evidence and bounded repair scope

1. run.stderr.log ends at run_same80_bits_r100.py:648:
   staged_fia._device_state does not exist. The established helper is in
   signed_int8_screen3_fia.py:85, already imported as screen_fia. Correct the
   scoring adapter to use the existing helper; do not change the public score,
   attack objective or source matching.
2. _recording_saver at lines508-527 installs a wrapper and RETURNS the original
   saver. Both _run_cpa and _run_fia assign that returned original back to the
   module's _save_checkpoint immediately, disabling the wrapper. Fix the call
   wiring so authenticated sidecars are actually saved, and restore the original
   saver once in finally. Do not rewrite checkpoint persistence infrastructure.
3. _score_checkpoint also expects top-level raw_image_sha256, while the actual
   staged_fia.optimize_until checkpoint writer records attack_state, RNG, horizon,
   log and iterations rather than that field. Check this concrete compatibility
   issue and validate restored pixels against the saved attack_state.S_hat bytes;
   never remove the no-mutation check. Test the real staged checkpoint schema
   through scoring, not only fixtures that invent the missing field.
4. Completed bit08/batch00 Naive/FIA is recorded as COMPLETED at25000 steps in
   naive_fia_attempts.json, with checkpoint SHA256
   b40e114e85317563996bad4b8e923c6ed1ed0548a3d0034ac9d54d154c65e6ff.
   Reverify that exact checkpoint hash against the pre-existing completion ledger,
   checkpoint schema/horizon/state, source manifest and input identity. Preserve
   its original bytes and completion record. A missing sidecar from defect2 must
   not be fabricated as an original record. If all original evidence authenticates
   it, write an explicitly dated recovery attestation linking the original ledger,
   source identity, old file hash and repaired deployment. Resume from the completed
   state for public scoring/finalization, without repeating25000 optimization steps.
   If any necessary evidence is absent/mismatched, state that precise limit; do not
   silently treat unauthenticated state as trusted or replace the candidate.
5. No live research worker remains (old PID14680 and child22048 exited code1).
   Reverify this and the supervisor locks before repairs/launch. Existing supervisor
   PID18648 is only the watchdog, currently NEEDS_REVIEW. Retain its policy and logs.

## Provenance and supervisor reconciliation

These are reviewed implementation repairs, not automatic retries of unreviewed
deterministic errors. Keep the old failed worker/stage records, original registration
and code identities byte-preserved. Record exact before/after code hashes, this
repair authorization and focused validation. Use the smallest explicit compatibility
or migration mechanism for the repaired deployment while keeping the ORIGINAL
scientific/input/config/seed identity and completed artifact lineage. The original
hash mismatch protections must still reject every unrelated change; do not simply
refresh all registered hashes, delete failures or disable integrity validation.

After the repair is validated, use a serialized, audited repair-resolution
transition to clear ONLY this reviewed deterministic blocker and permit the SAME
unfinished task under the existing supervisor. This is distinct from ordinary
--resume and USER_STOP. Do not turn all FAILED_ERROR/NEEDS_REVIEW cases into automatic
retry, alter a user stop, or race the live watchdog. A minimal supervisor repair
transition/allowlisted deployment record is authorized for this exact scope.
Restart/reload the task-specific watchdog if its own loaded code needs updating;
do not run two watchdogs or bypass the direct-launch guard.

Missing resource_measurement.json is not a reason to invent measurements or rerun
completed work. Recover only timings actually saved in checkpoint/log/result files;
mark missing peaks as unavailable for that attempt. Obtain forward measurements
from the next scheduled regular candidate, with disclosed provenance, no extra pilot.

## Acceptance and reporting

- Focused regression tests must exercise saver installation + original restoration,
  written sidecars, actual staged checkpoint -> unchanged public score, preserved
  pixel state, authenticated legacy completed checkpoint reuse with ZERO new attack
  steps, rejected corrupted/mismatched lineage, reviewed repair transition, and
  retained duplicate-worker protection. Use isolated roots and minimal fixtures.
- Complete the saved checkpoint's public scoring using the repaired actual path;
  this is approved unfinished work, not an extra experiment. Preserve selection rules.
- Resume the remaining run/gate/report through the single supervisor, preserving all
  prior artifacts and avoiding prepare/recollection/retraining. Report real worker
  PID/start, supervisor ownership, active automatic recovery and login persistence.
- Send one concise UTF-8 repair/restart report to the current Planner (or an exact
  newly evidenced blocker). Use a UTF-8 --message-file to avoid the garbled Korean
  in the prior report. End the model turn after verified launch; no repeated polling.

The user still expects the complete approved experiment and automatic restart.
The existing HKCU Run mechanism provides recovery after user login, not before
login; hang detection and watchdog self-recovery before the next login remain
unsupported and must be reported accurately. Do not request privilege escalation
or change account/security settings to remove these limits.
