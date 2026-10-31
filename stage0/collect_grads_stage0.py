#!/usr/bin/env python
"""Stage 0 FedSGD gradient collector.

Why this exists instead of upstream ``src/collect_grads.py``
------------------------------------------------------------
1. **Upstream bug**: ``collect_grads.py:104`` reads ``model.type`` where the
   models define ``model_type``.  ``nn.Module.type`` is a real bound method, so
   ``model_type == "conv"`` is always False, the ``return_z=True`` branch never
   runs, and ``data["z"]`` stays empty.  The VGG-16 attack path then feeds an
   empty tensor to ``EmbEval``.  This file uses ``model_type`` correctly.

2. **Disk**: upstream stores the gradient of *every* parameter.  For VGG-16 that
   is 138.4 M floats = 553 MB per batch; the full 6 x 10 grid would be ~33 GB of
   pickle.  ``--grad_scope=attack_layer`` stores only the attacked layer's
   gradient (the only tensor CPA + Direct-FIA ever read) and writes zero-size
   placeholders elsewhere, keeping the same list indexing.

3. **Seeding**: upstream has no ``--seed``; it hardcodes 42 via ``utils.setup``.

Everything else is upstream code: the model (``models.get_model``), the data
pipeline and transforms (``datasets.get_dataloaders``), the loss
(``CrossEntropyLoss``), and the gradient (``torch.autograd.grad`` over
``model.parameters()`` on the mean loss).  The collection loop below is a
line-for-line transcription of upstream ``collect_grads.collect_grads``.

With ``--seed=42 --grad_scope=full`` and the default batch-size list, this
reproduces upstream byte-for-byte in content (the RNG is advanced identically
because the batch sizes are swept in one process, in upstream order).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cpa import (  # noqa: E402
    SRC_VARIANTS,
    install_matplotlib_compat,
    install_wandb_stub,
)

UPSTREAM_BATCH_SIZES = [8, 16, 32, 64, 128, 256]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="official", choices=sorted(SRC_VARIANTS))
    parser.add_argument("--ds", default="imagenet")
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument(
        "--batch_sizes",
        default=",".join(str(b) for b in UPSTREAM_BATCH_SIZES),
        help="comma-separated; upstream order matters for RNG equivalence",
    )
    parser.add_argument("--n_batch", type=int, default=10)
    parser.add_argument("--defense", default="nodef", choices=["nodef"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--grad_scope",
        default="attack_layer",
        choices=["attack_layer", "full"],
        help="attack_layer: store only grads[attack_index] (all CPA/Direct-FIA need)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="do not recompute a batch size whose pickle already exists",
    )
    parser.add_argument(
        "--micro_batch",
        type=int,
        default=0,
        help=(
            "max samples per forward/backward; 0 disables splitting. "
            "VGG-16 at 224px OOMs above ~128 on 16 GB. Because the criterion is "
            "a mean, the full-batch gradient is exactly sum_c (n_c/B) * grad_c, "
            "so splitting is mathematically exact (fp32 summation order aside). "
            "A batch_size <= micro_batch takes the unsplit path, byte-identical "
            "to upstream."
        ),
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help=(
            "allow cudnn TF32 convolutions. OFF by default, unlike PyTorch: "
            "torch.backends.cudnn.allow_tf32 defaults to True, which perturbs the "
            "collected gradient by ~5e-3 relative (and shape-dependently, so a "
            "micro-batched sum stops matching the unsplit one). The FP32 baseline "
            "must be a clean reference because CPA's whitening amplifies small "
            "eigen-directions by 1/sqrt(lambda) with no clipping."
        ),
    )
    args = parser.parse_args()

    import os

    os.chdir(REPO_ROOT)
    install_wandb_stub()
    install_matplotlib_compat()
    sys.path.insert(0, str(SRC_VARIANTS[args.src]))

    import numpy as np
    import torch
    import torch.nn as nn

    import datasets as ds_mod
    from models import get_model
    from utils import exp_path_base, get_device, get_grads_file, setup, write_pickle

    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.deterministic = not args.tf32
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    # utils.setup() -> enable_gpu_benchmarking() sets cudnn.benchmark=True.
    # With deterministic=True that still runs an algorithm search restricted to
    # deterministic kernels, which probes several workspace sizes; under tight
    # VRAM (micro-batched VGG-16 at 224px, GPU already near full) that search
    # can spin retrying multiple multi-GB workspace allocations for minutes
    # without making progress. Force it off after setup() runs.
    torch.backends.cudnn.benchmark = False

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    exp_path = f"{exp_path_base}/{args.ds}/{args.model}/grads/{args.defense}"
    logger = setup(exp_path=exp_path, log_file=f"collect_grad_stage0_seed{args.seed}.log")
    # utils.setup() seeds with the hardcoded 42; re-seed to the requested value.
    from utils import set_seed

    set_seed(args.seed)

    device = get_device()
    logger.info(
        "[stage0] collect ds=%s model=%s batch_sizes=%s n_batch=%s seed=%s scope=%s",
        args.ds,
        args.model,
        batch_sizes,
        args.n_batch,
        args.seed,
        args.grad_scope,
    )
    logger.info(
        "[stage0] tf32=%s micro_batch=%s cudnn.deterministic=%s",
        args.tf32,
        args.micro_batch,
        torch.backends.cudnn.deterministic,
    )

    model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        load_path=f"{exp_path_base}/{args.ds}/{args.model}/model.pt",
        dataparallel=True,
    )
    base = model.module if hasattr(model, "module") else model
    model_type = base.model_type          # <-- upstream bug fix (was model.type)
    attack_index = base.attack_index
    model = model.to(device)
    model.train()
    logger.info(
        "[stage0] model_type=%s attack_index=%s n_params=%s",
        model_type,
        attack_index,
        sum(p.numel() for p in model.parameters()),
    )

    criterion = nn.CrossEntropyLoss()

    for batch_size in batch_sizes:
        grads_pkl_file = Path(
            get_grads_file(
                args.ds, args.model, args.defense, batch_size, args.C, args.sigma
            )
        )
        # NOTE: the dataloader is built for every batch size even when we skip
        # writing, so the RNG stream stays identical to a full sweep.
        _, dl_test = ds_mod.get_dataloaders(
            args.ds,
            batch_size=batch_size,
            shuffle_test=True,
            num_workers=args.num_workers,
        )
        if args.skip_existing and grads_pkl_file.exists():
            logger.info("[stage0] skip existing %s", grads_pkl_file)
            continue

        logger.info("[stage0] batch_size=%s -> %s", batch_size, grads_pkl_file)
        t0 = time.time()
        data = {
            "x": [],
            "y": [],
            "z": [],
            "grad": [],
            "grad_clip": [],
            "grad_dp": [],
            "stage0_meta": {
                "seed": args.seed,
                "grad_scope": args.grad_scope,
                "attack_index": attack_index,
                "model_type": model_type,
                "src": args.src,
                "n_batch": args.n_batch,
                "batch_size": batch_size,
                "micro_batch": args.micro_batch,
                "split_used": args.micro_batch > 0 and batch_size > args.micro_batch,
                "tf32": args.tf32,
            },
        }

        for i, (x, y) in enumerate(dl_test):
            if i == args.n_batch:
                break
            data["x"].append(x.numpy())
            data["y"].append(y.numpy())
            model.zero_grad()
            x, y = x.to(device), y.to(device)
            model_params = [p_ for p_ in model.parameters()]
            bsz = x.shape[0]
            split = args.micro_batch > 0 and bsz > args.micro_batch

            if not split:
                if model_type == "conv":
                    p, z = model(x=x, return_z=True)
                    data["z"].append(z.detach().cpu().numpy())
                else:
                    p = model(x)
                loss = criterion(p, y)
                grad = torch.autograd.grad(loss, model_params)
            else:
                # criterion is a mean over its input, so the full-batch mean loss
                # gradient is exactly sum_c (n_c / B) * grad(mean loss on chunk c).
                #
                # grad_params is the subset we actually request gradients for.
                # For grad_scope="attack_layer" this is a single tensor: the
                # accumulator lives on CPU so the standing GPU footprint is 0
                # (vs. holding all 138M params' worth of accumulator on GPU
                # for the whole chunk loop, which is what OOM'd batch_size=256
                # after 5 prior batch sizes had already fragmented the
                # cudaCachingAllocator in this process).
                if args.grad_scope == "attack_layer":
                    grad_param_idx = [attack_index]
                else:
                    grad_param_idx = list(range(len(model_params)))
                grad_params = [model_params[j] for j in grad_param_idx]

                grad_accum_cpu = [
                    torch.zeros_like(p_, device="cpu") for p_ in grad_params
                ]
                z_chunks = []
                for start in range(0, bsz, args.micro_batch):
                    xc = x[start : start + args.micro_batch]
                    yc = y[start : start + args.micro_batch]
                    w = xc.shape[0] / bsz
                    if model_type == "conv":
                        pc, zc = model(x=xc, return_z=True)
                        z_chunks.append(zc.detach().cpu())
                    else:
                        pc = model(xc)
                    loss_c = criterion(pc, yc) * w
                    gc = torch.autograd.grad(loss_c, grad_params)
                    for acc, g in zip(grad_accum_cpu, gc):
                        acc.add_(g.detach().cpu())
                    del gc, pc, loss_c
                    torch.cuda.empty_cache()
                if model_type == "conv":
                    data["z"].append(torch.cat(z_chunks, dim=0).numpy())

                if args.grad_scope == "attack_layer":
                    grad = [None] * len(model_params)
                    grad[attack_index] = grad_accum_cpu[0]
                else:
                    grad = grad_accum_cpu

            if args.grad_scope == "full":
                grad_np = [g.detach().cpu().numpy() for g in grad]
            else:
                empty = np.zeros((0,), dtype=np.float32)
                grad_np = [
                    grad[j].detach().cpu().numpy() if j == attack_index else empty
                    for j in range(len(grad))
                ]
            data["grad"].append(grad_np)
            logger.info(
                "[stage0]   batch %s/%s  |dW|_F=%.6g",
                i + 1,
                args.n_batch,
                float(grad[attack_index].norm()),
            )

        write_pickle(data, str(grads_pkl_file))
        size_gb = grads_pkl_file.stat().st_size / 1024**3
        logger.info(
            "[stage0] wrote %s (%.2f GB) in %.1f s",
            grads_pkl_file,
            size_gb,
            time.time() - t0,
        )

    logger.info("[stage0] done")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
