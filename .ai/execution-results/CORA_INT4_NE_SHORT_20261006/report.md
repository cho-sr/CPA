# CORA_INT4_NE_SHORT_20261006

Status: FAILED, partial separation verified. Worker exited1.

Member0: CPA500/500 completed; FIA0/500 completed. Member1 was not started. The saved target has shape16x25088 and is finite. Public input and registered code hashes match.

Negentropy was active in all six saved loss logs: first -0.052121773, last -0.042632557. The contrast was included in the total loss; this does not establish reconstruction quality.

FIA stopped during its first backward call because adaptive_avg_pool2d_backward_cuda does not support the strict deterministic-algorithm mode retained by this task's short-run adapter. The established FIA helper uses seed_all(...), which selects warn_only=True; that runtime setup was omitted here. This is an execution-adapter error, not evidence that the enabled negentropy failed.

No final pixels, public candidate selection, private evaluation, PSNR/SSIM/LPIPS or comparison image were produced. No optimization was restarted. Original worker failure is preserved in worker_failure_original.json.
