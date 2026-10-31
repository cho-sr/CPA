run_tag="m6_round_spectrum_decomposition"

# n=32 Prediction Reconciliation

Matched condition check: m5 seed0/batch0/T=1 predicts `24` and m6 round0/seed0/batch0 predicts `24`.

The apparent 24/32 vs 21.2/32 mismatch is an aggregation-axis difference: 24 is the single seed0,batch0 condition, while 21.2 is the m6 mean over five measurement batches at seed0 (`24,25,22,21,14`).
