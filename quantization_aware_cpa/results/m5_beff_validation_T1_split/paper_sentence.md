run_tag="m5_beff_validation_T1_split"

예측식은 96 조건 중 81 에서 정확했다 (MAE 0.219). FedSGD 에 해당하는 T=1 부분격자에서는 30/36 가 정확했다.

The 30/36 denominator is the core per-row grid n in {16,32,64} x bits in {2,3,4,8} x seed in {0,1,2}. Including the extra 4-bit per-tensor/per-bucket granularity checks, T=1 is 39/48.
