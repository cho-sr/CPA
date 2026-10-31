# CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916

Status: **COMPLETED_WITH_REVIEW**.

All registered stages and the one offline four-set evaluation completed with exit status 0. The comparison remains **NEEDS_REVIEW** because one archived-control LPIPS reproduction difference exceeds the predeclared absolute 2e-8 tolerance. This is a reporting review flag, not a rerun trigger.

## Scope and execution

- One registered pipeline only: CPU FP32 FastICA + 1,000 prior-refinement steps, CUDA Direct FIA 25,000 steps, and CUDA actual-image cell refinement 100 steps.
- No extra arm, seed, tuning, continuation, generator, private labels, true updates, or private feedback.
- Wrapper process exit: 0; model calls while running: 0; total wall time: 2330.78867509996 s / 2700 s cap.
- Prelaunch validation: 7 passed; compile validation is recorded in registration.json and registration_initial.json.
- Task-local files added for this run: isolated experiment/evaluation orchestration, focused tests, wrapper, and reports. Historical code/results and protected/shared sources were retained.

## Stage records

| Stage | Status | Exit | Wall seconds | Exact work |
|---|---|---:|---:|---|
| SEPARATE | COMPLETED | 0 | 33.069726 | FastICA converged in 84 iterations; prior refinement 1000 |
| INVERT | COMPLETED | 0 | 2160.407650 | Direct FIA 25000 |
| REFINE | COMPLETED | 0 | 2308.128908 | actual-update cell refinement 100 |
| PRIVATE_EVALUATION | COMPLETED | 0 | 2330.781256 | offline evaluation of four frozen sets |

Separate emitted native estimated order S_hat and exact target_z=abs(S_hat). The refinement preflight reports feature grad/no-grad equality and terminal delta equality for all six FC tensors. The cell result recorded peak CUDA allocated memory of 3965342208 bytes under the registered 12 GiB cap.

## Public gate and endpoint feasibility

Public gate: **PUBLIC_GATE_PASSED** before private loading. Protected/shared/public-input integrity: **unchanged**. The full per-tensor profiles are retained in quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/comparison.json and final_audit.json.

| Set | Exact feasible | Code mismatches | Fraction | Bounded-cell violations | Fraction | Cell loss |
|---|---:|---:|---:|---:|---:|---:|
| original_separator_direct | false | 1210569 | 0.00979085277681 | 1210573 | 0.00979088512805 | 22.9916826474 |
| original_separator_cell | false | 996857 | 0.00806239060023 | 996857 | 0.00806239060023 | 0.204078117165 |
| combination_separator_direct | false | 1225207 | 0.00990924214821 | 1225212 | 0.00990928258726 | 22.8891772005 |
| combination_separator_cell | false | 995058 | 0.00804784062898 | 995058 | 0.00804784062898 | 0.192871173711 |

Exact feasibility is defined as zero fixed-observed-scale code mismatches and zero bounded-cell violations across all six tensors. The new cell endpoint is therefore not exactly feasible; this is reported separately from image quality.

## Private four-set evaluation

Private truth was loaded only after the public gate. The evaluator consumed x only; labels_consumed=false, true_updates_consumed=false, and no private feedback entered the attack. Hungarian pixel-MSE correspondence was computed once per candidate and reused for all listed metrics and the aligned figure. All 16 images were included.

| Set | MSE | PSNR dB | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| original_separator_direct | 0.0781093468343797 | 11.5173321456311 | 0.284893989708307 | 0.822924602776766 |
| original_separator_cell | 0.078160877077264 | 11.52174236431 | 0.282700870678415 | 0.808156996965408 |
| combination_separator_direct | 0.0776844093863262 | 11.5249464319988 | 0.29322348980258 | 0.809157457202673 |
| combination_separator_cell | 0.077736043680349 | 11.5214056622864 | 0.284907516904722 | 0.800327986478806 |

Predeclared contrasts are stored in comparison.json: combination direct vs original direct, combination cell vs combination direct, and combination cell vs original cell. They are paired descriptive results for this fixed batch and do not support a generalization claim.

## Control reproduction review

The original direct and original cell controls were re-evaluated and their candidate hashes matched the registered archived hashes. The cell control reproduces within tolerance. The direct control differs in LPIPS by -2.2351741790771484e-08 against the 2e-8 bound, so the overall control reproduction status is **NEEDS_REVIEW**; metrics are preserved without loosening the bound or rerunning.

## Artifact paths

- Execution summary: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916/execution_summary.json
- Stage execution: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916/stage_execution.json
- Final audit: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916/final_audit.json
- Private comparison: quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/comparison.json
- Private report: quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/report.md
- Figure: quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/all16_comparison.png
- Wrapper-generated report preserved as .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916/report_wrapper_generated.md

No further execution is authorized or implied by this report.
