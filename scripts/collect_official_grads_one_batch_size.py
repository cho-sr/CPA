#!/usr/bin/env python
"""Collect one official CPA gradient pickle for a selected batch size."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from run_official_cpa import install_torchtext_stub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", default="imagenet")
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_batch", type=int, default=10)
    parser.add_argument("--defense", default="nodef", choices=["nodef", "dp"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    official_src = repo_root / "official_cpa" / "src"
    install_torchtext_stub()
    sys.path.insert(0, str(official_src))

    from collect_grads import apply_dp_noise  # noqa: WPS433
    from datasets import get_dataloaders  # noqa: WPS433
    from models import get_model  # noqa: WPS433
    from utils import exp_path_base, get_device, get_grads_file, setup, write_pickle  # noqa: WPS433

    grads_pkl_file = Path(
        get_grads_file(args.ds, args.model, args.defense, args.batch_size, args.C, args.sigma)
    )
    if args.reuse_existing and grads_pkl_file.exists():
        print(f"Reusing existing gradient pickle: {grads_pkl_file}")
        return

    exp_path = f"{exp_path_base}/{args.ds}/{args.model}/grads/{args.defense}"
    logger = setup(exp_path=exp_path, log_file=f"collect_grad_{args.batch_size}.log")
    logger.info(
        "Collecting official gradients with %s model on %s batch_size=%s n_batch=%s",
        args.model,
        args.ds,
        args.batch_size,
        args.n_batch,
    )
    device = get_device()

    dataparallel = False if args.defense == "dp" else True
    model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        load_path=f"{exp_path_base}/{args.ds}/{args.model}/model.pt",
        dataparallel=dataparallel,
    )
    base_model = model.module if hasattr(model, "module") else model
    model_type = getattr(base_model, "model_type", getattr(model, "model_type", None))
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    _, dl_test = get_dataloaders(
        args.ds,
        batch_size=args.batch_size,
        shuffle_test=True,
        num_workers=0,
    )
    data = {"x": [], "y": [], "z": [], "grad": [], "grad_clip": [], "grad_dp": []}

    for i, (x, y) in enumerate(dl_test):
        if i == args.n_batch:
            break
        logger.info("Collecting batch %s/%s", i + 1, args.n_batch)
        data["x"].append(x.numpy())
        data["y"].append(y.numpy())
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        if model_type == "conv" or args.ds == "imdb":
            p, z = model(x=x, return_z=True)
            data["z"].append(z.detach().cpu().numpy())
        else:
            p = model(x)
        loss = criterion(p, y)

        if args.defense == "dp":
            loss.backward()
            sample_grad_list = [p.grad_sample.detach() for p in model.parameters()]
            grad = apply_dp_noise(sample_grad_list, args.C, args.sigma)
        else:
            model_params = [p for p in model.parameters()]
            grad = torch.autograd.grad(loss, model_params)

        data["grad"].append([g.detach().cpu().numpy() for g in grad])

    write_pickle(data, str(grads_pkl_file))
    logging.getLogger().info("Done. Data saved in %s", grads_pkl_file)
    print(f"Saved gradient pickle: {grads_pkl_file}")


if __name__ == "__main__":
    main()
