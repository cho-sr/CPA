# CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918

Status: **COMPLETED_AFTER_REPORTING_REPAIR**.

The saved public continuation completed before the wrapper stopped on a reporting-only truth-schema defect. The original failed wrapper receipt and logs are preserved as wrapper_failure_* files. The evaluator was corrected to accept the registered private truth bundle (x, z, labels) while consuming x only; no optimizer step was repeated.

## Public completion

- Public gate: **PUBLIC_GATE_PASSED**; 13 frozen candidate sets authenticated before private loading.
- Frozen public selection: [1, 7, 5].
- Member01 recovery export: 0 optimizer steps; member07 and member05: 20,000 additional steps each.
- Cumulative FIA updates: 125,000; actual-image cell feasibility: NOT_EVALUATED.

| Set | MSE | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| center_5000 | 0.0813593276548 | 11.3829456329 | 0.166010470604 | 0.56794205308 |
| member_00_5000 | 0.0808870750928 | 11.4089585585 | 0.163051420745 | 0.57916348055 |
| member_01_5000 | 0.0807361178772 | 11.413067876 | 0.166375697439 | 0.570748103783 |
| member_02_5000 | 0.0816752318262 | 11.345904126 | 0.164436735279 | 0.563299030066 |
| member_03_5000 | 0.080007006237 | 11.4267131335 | 0.169834198187 | 0.565670616925 |
| member_04_5000 | 0.0810137976994 | 11.3708314272 | 0.165801954931 | 0.589287782088 |
| member_05_5000 | 0.0811735404315 | 11.3759009851 | 0.167683936261 | 0.564295986667 |
| member_06_5000 | 0.0813110105426 | 11.4016849179 | 0.160407519594 | 0.579947270453 |
| member_07_5000 | 0.0810060160274 | 11.366671858 | 0.166934637689 | 0.574944114313 |
| center_25000 | 0.0706620541313 | 12.0569057933 | 0.292355354623 | 0.559406101704 |
| rank1_member_01_25000 | 0.0701470816373 | 12.0567617973 | 0.307018888898 | 0.555841319263 |
| rank2_member_07_25000 | 0.0703305734302 | 12.0440286399 | 0.305054534773 | 0.554504306987 |
| rank3_member_05_25000 | 0.0704995550494 | 12.0291946019 | 0.305001579278 | 0.553453439847 |

Primary contrast (rank1 member01 minus center25000):

| MSE | PSNR dB | SSIM | LPIPS |
| ---: | ---: | ---: | ---: |
| -0.000514972494054 | -0.000143996046578 | 0.0146635342753 | -0.00356478244066 |

Canonical runtime: seed 42, one CPU thread. Matching: one pixel-MSE Hungarian assignment per candidate reused for MSE/PSNR/SSIM/LPIPS and aligned figures.
Private boundary: x only; labels and true updates were not consumed by attack or evaluator. Sender true-update consumption is reported separately.

Figures: contact_sheet_5000.png, contact_sheet_25000.png.
