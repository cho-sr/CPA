# Top-k10 random filling: same80 CPA + FIA

The exact five packets previously attacked with zero fill are reused. Same original80 images, n16, round0 frozen-conv VGG16, all-classifier local SGD lr0.001 for10 steps. No new victim training or update collection.

Only discarded coordinates are replaced by independent U(-tau,tau) samples, where tau is the minimum retained absolute value in the packet. Retained values remain bit-exact FP32. Each batch has20 fills, each with original CPA25k and FIA5k. The best3 observable update-consistency scores continue from the same Adam/cosine state to TOTAL25k; the cosine horizon is25k from the first step. CPA seeds and FIAseed42 match the baseline for every candidate.

Ranking replays the same10-step classifier SGD on all16 reconstructed images, with labels predicted by the public round0 model, then applies the same FC1 Top-k codec. Relative L2 to the received decoded packet is minimized, with candidate index breaking ties. This observable proxy does not guarantee better image metrics. True labels/images/features are unavailable to construction and ranking. Final selection is one whole16-image candidate per batch, among the3 finalists, using the same proxy. Only then does the unchanged signed-source absolute-cosine Hungarian evaluation run.

| Method | LPIPS lower | PSNR dB higher | SSIM higher |
|---|---:|---:|---:|
| Zero fill (1 candidate, FIA25k) | 0.632326 | 21.5305 | 0.263122 |
| Uniform random fill (20 at5k, top3 to25k) | 0.629989 | 21.5104 | 0.261004 |

Same actual FC1 communication: 269,746,320 bytes across5 packets, 86.8750% fewer bytes than same-shape denseFP32. All100 fills and115 replay scores are attacker-local computation with zero extra transmitted bytes. No full-model traffic or training-utility claim.

The user-requested attack budget differs: per batch the baseline has CPA25k+FIA25k, while random filling totals CPA500k+FIA160k and23 ranking replays. Consequently this comparison does not isolate filling from increased search compute. No extra zero-fill restart control was run.

All3 final candidate reconstructions and all20 screen reconstructions remain saved per batch. The canonical all80 metrics/PNG use only the predeclared observable final selection, never the best LPIPS/SSIM/PSNR per image.

![All80 original and reconstruction pairs](all80_original_vs_reconstruction_16x9.png)
