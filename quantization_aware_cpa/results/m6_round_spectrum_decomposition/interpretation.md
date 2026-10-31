run_tag="m6_round_spectrum_decomposition"

# Round Spectrum Decomposition

No shrinking-scale cancellation is observed: sigma2, mp_edge, and lambda_max rise mildly while their ratio is nearly preserved. Because the measurement loss changes only slightly, hold the learning-stage-insensitivity paragraph.

Round 0 to 100 changes:
- train loss: 0.7276 -> 0.2448
- measurement loss: 1.1856 -> 1.1529
- sigma2: 1.330e-07 -> 1.371e-07
- mp_edge: 2.622e-07 -> 2.703e-07
- lambda_max: 3.061e-04 -> 3.218e-04
- lambda_max/mp_edge: 1167.2 -> 1190.4
