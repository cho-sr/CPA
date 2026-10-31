# FedAvg naive CPA + FIA (direct upstream)

Completed all80 originals: VGG16, n16, T10, same cached actual-update dither4 observations.

| Condition | LPIPS | PSNR | SSIM | Embedding cos>0.9 |
|---|---:|---:|---:|---:|
| Direct upstream CPA + FIA | 0.556121 | 21.937 | 0.2980 | 53/80 |
| Earlier ported UIA4bit | 0.558584 | 21.926 | 0.2976 | 53/80 |

No LS-A calculation, debias, MP diagnostics/truncation, Stage2, refinement, oracle fills or new client training.
The original CocktailPartyAttack and Direct methods are invoked directly. Original whitening uses float32 covariance/(d-1) and eig; it differs from the earlier port's float64 covariance/d and eigh.
FIA takes abs(S_hat) in estimated source order. Hungarian correspondence is applied after FIA only; this evaluation adapter differs from upstream's greedy correspondence before FIA and changes the assignment of initial image noise to sources.
The backend and evaluation adapter both differ from the earlier port. Any performance change is not an isolated effect of removing LS-A.
Observation is the same per-row 4-bit-interval uniform-dither model, not a transmitted integer 4-bit codec.

![All80 landscape](all80_reconstruction_landscape.png)
![Random12 not cherry-picked](random12_not_cherry_picked.png)
