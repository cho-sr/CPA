# FP32 run stopped before ICA

The saved traceback places the failure in whiten_observation(X,16), before FastICA, GD-ICA or FIA. The input/source provenance gate passes. No final candidate or iteration logs exist. Actual optimization counts: FastICA0, GD-ICA0, FIA0; configured budgets were4000/4000/25000.

Error: observation has insufficient usable covariance rank for source_count. The existing guard rejects nonfinite selected eigenvalues or any selected absolute eigenvalue at or below1e-12. Eigenvalues were not saved; this report does not establish the mathematical rank or whether scaling, numerical precision or data structure triggered the guard.

No reconstruction metrics are available. This execution failure does not establish that FP32 updates cannot be reconstructed. No whitening settings were changed and no optimization was restarted.
