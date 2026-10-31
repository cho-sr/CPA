# Short INT4 CoRA with negentropy restored

TASK_ID: CORA_INT4_NE_SHORT_20261006
PLAN_VERSION: 1
STATUS: FAILED
ASSIGNED_ROLE: CODEX_EXECUTOR
CLAIMED_BY: CURRENT_CONVERSATION:01a110c4-3fad-7d53-9951-d66ff4af0c6b
USER_EXECUTION_AUTHORIZATION: 짧게 돌려볼수있나 4비트로; preceding direct assignment: 걍 negentropy 켜줘 너가

One short run only: archived single-client INT4 B16/T10 observation; frozen VGG16 Conv, all FC local SGD0.001, seed42. Two bounded-cell candidates with registered seeds1234/1001237; exact public codes and per-tensor scale preserved. Each uses restored upstream log-cosh NE (ne1), Adam0.001, decor5.3/T7.7/nv0.13/l1=5/TV0.1, CPA500 steps; then unchanged Direct FIA500 steps, Adam0.1, cosine scheduler, fi1/TV0.1. All16 sources/images kept. Rank final candidates by public feature-cosine distance+0.1 normalized TV, tie by member index. Only after both attacks and public selection finish, evaluate the winner on all16 images with frozen pixel-MSE Hungarian pairing and PSNR/SSIM/VGG LPIPS. No new victim training, private feedback, additional arm, restart or success cutoff.

Worker: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CORA_INT4_NE_SHORT_20261006\run_short.py run
Public: C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results\cora_int4_ne_short_20261006
Private: C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\cora_int4_ne_short_20261006
Record: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CORA_INT4_NE_SHORT_20261006\execution_record.json
Use one hidden persistent worker and one verified OS-exit callback to this same conversation. End the model turn after launch; no progress polling.

## Verified failure (2026-10-06)

Member0 completed CPA500 with active negentropy; FIA failed before its first optimizer step due to strict deterministic CUDA adaptive pooling. Member1 did not start. No candidate selection, reconstruction pixels or private quality metrics. No restart. Reporting audit: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CORA_INT4_NE_SHORT_20261006\completion_verification.json
