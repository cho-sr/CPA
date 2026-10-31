# CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917

Status: **PARTIAL / FAILED_BEFORE_SENDER_COMPUTATION**.

The registered wrapper exited with code 1 after registration. The sender
subprocess failed before entering `collect()` because its CLI dispatcher passed
the argparse-only `command` field into `collect(**vars(args))`:

`TypeError: collect() got an unexpected keyword argument 'command'`

Completed stages:

- focused validation: 6/6 INT8 tests passed; `py_compile` and all three CLI help checks passed;
- registration: completed, exit 0;
- sender process: started, exit 1 before raw-update deserialization.

No INT8 public packets, randomized matrices, ICA calls, FIA runs, private
evaluation, metrics, or scientific conclusion were produced. Therefore
`scientific_quality` is `NOT_EVALUATED`, `true_updates_consumed` is false for
this failed execution, and the archived raw update, old INT4 artifacts, and
all historical results remain unchanged. No retry, arm, seed, optimizer step,
or condition change was performed.

Preserved evidence:

- `execution_summary.json`: wrapper status and stage exit records;
- `01_register.log` and `02_sender.log`: exact subprocess evidence;
- `registration.json`: pre-execution scope registration;
- `integrity.json`: input/checkpoint hashes and retained artifacts;
- `failure_analysis.json`: final audit of the stopping point.

The isolated implementation has a routine CLI repair remaining, but this
authorized one-time real execution is closed at the recorded failure and was
not restarted.
