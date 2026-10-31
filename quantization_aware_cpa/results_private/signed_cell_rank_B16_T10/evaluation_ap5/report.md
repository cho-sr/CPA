# Signed INT4 naive / external cell-rank comparison

[FACT] evaluation_status=COMPLETED; plot_status=COMPLETED.
[FACT] Both completed arms use the original CPA25k, abs source adapter, and Direct FIA25k.
[FACT] All 16 final candidates are evaluated label-free with one pixel-MSE Hungarian matching per arm.
[FACT] Cell/code and scale checks describe the inferred update input; they do not describe a reconstructed local update.
[FACT] Cell-rank minus naive mean deltas: {"cell_rank_minus_naive_mse": -0.00017946736216596648, "cell_rank_minus_naive_psnr_db": -0.01280671675179157, "cell_rank_minus_naive_ssim": -0.0032370245524492858, "cell_rank_minus_naive_lpips": -0.028756409883499146}
[FACT] Evaluation-only true-G errors: {"naive_sq": {"status": "COMPLETED", "arm": "INT4_naive_sq", "attack_layer": "classifier.0.weight", "estimate_input_sha256": "815f28e872de6f1b1d565cb413f26e958fa66758cd6fbbaad85758f77552f69c", "true_g_source": "raw_updates.pt", "delta_convention": "theta_after10_minus_theta_start", "absolute_l2_error": 0.03371233328273856, "reference_l2_norm": 0.046040659538687155, "relative_l2_error": 0.732229590551601, "relative_l2_error_status": "COMPLETED", "selection_used": false}, "cell_rank_V": {"status": "COMPLETED", "arm": "INT4_cell_rank_V", "attack_layer": "classifier.0.weight", "estimate_input_sha256": "f323e0a88ef9f7d9c38c5fe15e25683561c21aa685e5956d524b8a592424a6b5", "true_g_source": "raw_updates.pt", "delta_convention": "theta_after10_minus_theta_start", "absolute_l2_error": 0.02917478452002653, "reference_l2_norm": 0.046040659538687155, "relative_l2_error": 0.6336743394284235, "relative_l2_error_status": "COMPLETED", "selection_used": false}}
[UNKNOWN] No threshold, significance, or generalization is established by one batch/seed.
