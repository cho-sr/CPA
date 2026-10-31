# Actual launch blocker found on user status request

User asks "언제 실행해" after authorizing the fresh16/R8 v4 scope.
Read-only inspection at2026-09-21 08:08KST found:
- No running sparse experiment Python worker and no fresh16 output/start record.
- Active handoff remains v2/all80/R8. Existing Luna is active in the turn that
  began07:56:15KST, processing the first R8 amendment.
- Its saved received-user-message history contains v1 and v2 only; the accepted
  v3/v4 queue messages have not yet appeared as model input in that history.
- Latest recorded tool work was creation of run_paper_sparse_same80_r8.py.
Therefore v4 was queued but had not yet been acted on, and no valid fresh16
start time or calendar completion time was available.

The installed `codex queue` exposes queueing only, no immediate steering flag.
No verified shared app-server control endpoint is available for this terminal;
no replacement AI was started and no live thread store was edited to bypass it.
Do not resend v4 repeatedly or confuse transport acceptance with model receipt.

Asked the user to interrupt only the ongoing Luna model response using Esc and
enter the authoritative v4 message directly in that existing terminal. This is
an operational message-delivery prerequisite, not repeat experiment approval.
Existing background tasks/files must remain safely managed by that Executor.
Once v4 is directly received, older queued v2/v3 notices are superseded and
must never revert the16-image/fresh8-candidate task or launch additional runs.

Expected fresh compute remains approximately8--9h from actual successful launch,
including fresh naive and random arms. Setup/transition time is additional.
