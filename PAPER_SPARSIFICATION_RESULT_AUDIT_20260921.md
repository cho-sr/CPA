# Sparsification random reconstruction artifact audit

User request: check whether the figure's `Not run` entries really have no saved results.
Role: CODEX_PLANNER; existing-file inspection and manuscript correction only.

## Finding

`Not run` was too strong. Completed ImageNet/VGG16 randomized CoRA runs and selected reconstruction artifacts exist at **50% and 30% retention**. They use a different protocol from the current same80/B16/T10 comparison. A 20%-retention randomized reconstruction was not located; saved 20% naive/zero-fill results do exist.

| Retention | Saved randomized result | Fits the current matched comparison? |
|---|---|---|
| 50% | FedAvg-context CoRA, selected member 000; additional single-client and aggregate variants also exist | No: different batch, local steps, checkpoint, sampler and search budget |
| 30% | FedAvg-context CoRA, selected member 004; an additional single-client variant also exists | No: same mismatches |
| 20% | No randomized reconstruction found in the inspected result records | No matched result found; naive results exist |
| 10% | `cpa_fia_random_topk10_same80_R20_S5k_K3_F25k` | Yes; already used in the manuscript |

## Directly verified 50% and 30% evidence

The FedAvg run root is:

`cora_fedavg_single_client_cpa/outputs_v1/imagenet/vgg16/fedavg_single_client_clients3_round20of20_spc24_target0_local1_e920201510be97a8/global_topk/`

For each `keep_0p5` and `keep_0p3`, `R8_seed1234_model_pred/manifest.json` and `consistency_member_ranking.csv` agree on the selected member: 0 and 4 respectively. Each points to the following reconstruction directory:

- [50% selected candidate](sparsified_update_cpa/outputs_v1/author_runs/imagenet/vgg16/attack/cp_direct/nodef/cora_fedavg_single_client_cpa_keep0p5_n24_steps1_trials1_R8_seed1234_m000_uia)
- [30% selected candidate](sparsified_update_cpa/outputs_v1/author_runs/imagenet/vgg16/attack/cp_direct/nodef/cora_fedavg_single_client_cpa_keep0p3_n24_steps1_trials1_R8_seed1234_m004_uia)

Both contain `batch_000_fi.pt`, `reconstructions.pkl`, `manifest.json`, `losses.jsonl`, and `metric_summary.csv`. These are actual reconstruction outputs, not just planned configurations. Existing full image reports were opened and visually inspected; their headings identify the same selected member and FIA 25k endpoint:

- [50% selected reconstruction PNG](cora_fedavg_single_client_cpa/outputs_v1/png_keep50_recon/slide/all_batches_original_reconstruction_metrics_keep50.png)
- [30% selected reconstruction PNG](cora_fedavg_single_client_cpa/outputs_v1/png_keep30_recon/slide/all_batches_original_reconstruction_metrics_keep30.png)

These historical PNGs contain threshold annotations. They are evidence of saved outputs and are not inserted into the paper.

## Why these outputs do not fill the current empty cells

| Setting | Current paper's sparse comparison | Located FedAvg-context 50%/30% runs |
|---|---|---|
| Batch size / local steps | B=16 / T=10 | B=24 / T=1 |
| Checkpoint | Starting round 0 VGG16 | Before the observed round 20 |
| Victim data | Same cached80, five original batches | Separate victim batch; displayed source identities differ |
| Random candidates | 20 | 8 screened |
| Continued candidates | 3 | 5 |
| Missing-value sampler | `uniform_tau_pcg64_v1` | `retained_rescaled` |
| Final candidate | Best observable relative-L2 score among three finalists | Observable consistency selection among five finalists |

Current settings come from `quantization_aware_cpa/configs/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k.json`, which explicitly records authorization and execution for **Top-k10 ONLY**. The selected older run manifests record B24, local_steps1, their sampler and checkpoint. The old experiments are not merged into the matched table or image panel.

## Search coverage and correction

The file inventory covers 11,060 accessible JSON/CSV/Markdown/log/tensor/image/pickle artifacts and identifies 4,223 sparsification-related paths. It includes hidden and ignored result files, the `cora_*` families, `sparsified_update_cpa`, `quantization_aware_cpa`, exports, and copied-source folders. Dataset directories, compiler dependencies, git metadata, and temporary upstream/pytest workspaces were excluded. The only remaining enumeration error was `.pytest_cache`, not a result directory. The inventory is saved at `paper_assets/sparsification_search_inventory.json`.

A separate content search for 20% retention found metadata for the original naive n64 study, the same80 zero-fill runs, configs, exports and a copied source tree; none identified a randomized 20% reconstruction. No matched same80/B16/T10/R20 random outputs at 50%, 30% or 20% were located. This is an artifact-availability finding, not proof that no such experiment has ever run elsewhere.

The current figure's `Not run` text is replaced with **N/A**, and the caption specifies that this means an unavailable result **for the current matched protocol**. Both PNG/PDF assets and the compiled manuscript are updated. Numerical cells stay unfilled; no attacks, training, rematching or metric recomputation are performed.
