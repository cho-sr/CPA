# CIFAR-100 qualitative examples for the manuscript

The user selected display examples **11 and 24** from `cifar100_png` after
inspection, replacing the initial editorial suggestion of example03.
These are one-based positions in the supplied 24-image contact sheets,
not independently verified CIFAR-100 dataset sample IDs or class labels.

## Figure assets

- `cifar100_quantization_selected_examples11_24.png` and `.pdf`: CIFAR-only panel.
- `quantization_cifar100_imagenet.png` and `.pdf`: manuscript Figure1, adding
  the two CIFAR rows above the unchanged ImageNet examples0/1.
- `cifar100_selected_examples11_24/`:22 exact80x80 PNG tiles, comprising
  two originals and Naive/Random pairs for every bit width8/7/6/5/4.
- `cifar100_selection_manifest.json`: source hashes, crop coordinates,
  preserved assignments, selection disclosure and output hashes.
- Rebuild with `C:/miniconda/envs/DLG/python.exe paper_assets/build_cifar100_selected_panel.py`.

Image11 is batch01/column03; image24 is batch02/column08, with batch indices
zero-based and columns one-based. Every original is identical across the
five bit widths. All360 source tiles (24 images x3 methods x5 bits) match
their respective batch-sheet copies pixel-for-pixel. Cropping only extracts
whole tiles from the contact sheet; image content is not trimmed or enhanced.
The combined PDF reuses the previous ImageNet tensors and exact saved
candidate assignments. Rendering uses no smoothing and no averaging.

## Interpretation and scope

This is a **post-hoc qualitative selection**, not a preregistered or random
sample and not a numerically established optimum. The intention is to show
low-bit Naive/Random differences. Detail remains limited, and superiority
must not be asserted uniformly across all bit widths or the full24 images.
Both methods are shown at all five bit widths, including conditions with
less favorable comparisons. The manuscript caption and text disclose this.

The source folder contains supplied PNGs and an index, not the underlying
checkpoints, per-image metrics, candidate selections or matching records.
Their saved displayed pairing is preserved, not newly established. No new
attack, metric computation, Hungarian assignment or random candidate search
was run. Existing aggregate table values were not edited or regenerated.

Original ImageNet-only assets and the initial example03 proposal are retained
as history; neither the reserved CIFAR footer nor example03 is used in the
updated Figure1. The original manuscript is backed up as
`test_cora_before_cifar100_examples11_24.tex` in this directory.

## Sparsification companion panel

The subsequent user request adds the same CIFAR examples11/24 to the
sparsification figure, without selecting different examples for retention.
`sparsification_cifar100_imagenet.png` and `.pdf` combine these with the existing
ImageNet examples0/1, showing keep50/30/20/10 and Naive/Random pairs.
The standalone CIFAR panel is `cifar100_sparsification_examples11_24.png`.
Run `build_cifar100_sparse_panel.py` to rebuild; provenance and validation are
in `cifar100_sparse_manifest.json`. All288 full-sheet/batch-sheet tile pairs
match, and all24 originals match the quantization originals.

CIFAR Random is the supplied R8 result. ImageNet remains the matched
same80/B16/T10/R20 study, with Random at keep10 only and six N/A tiles at
keep50/30/20. Fresh16 partial results and incompatible older experiments are
not substituted. The original ImageNet-only asset and all metric tables
remain unchanged. The manuscript Figure2 now uses the combined asset.

## Compact, caption-led layout

Both figures now use the86mm single-column width. Within each sample pair,
Naive is the upper row and Random the lower row; the left column contains
the original. CIFAR occupies the top four rows and ImageNet the bottom four.
Image numbers, dataset headings, method names and the Original heading are
removed from the bitmap/vector panels. Only condition headings and N/A
placeholders remain. Captions explain the ordering and selection limitations;
source identities remain in manifests. No reconstruction pixels or selections
were changed by this typesetting revision.
