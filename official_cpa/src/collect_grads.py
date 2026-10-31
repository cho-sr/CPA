#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import torch.nn as nn
import torch
import submitit
import logging
import os

try:
    from opacus.grad_sample import GradSampleModule
except ImportError:
    GradSampleModule = None

JOB_NAME = "collect_grads"

from utils import (
    get_device,
    exp_path_base,
    setup,
    write_pickle,
    submitit_logdir,
    get_grads_file,
    slurm_partition,
)
from datasets import (
    get_dataloaders,
    ds_dict,
)
from models import get_model, model_class_dict


def apply_dp_noise(sample_grad_list, C=1, sigma=0):
    # sample_grad_list: list of sample_wise_gradients for each parameter, where each gradient has dimensionality: batch_size x param_size
    # C: L2 norm for clipping
    # sigma: stddev of noise
    batch_size = sample_grad_list[0].shape[0]
    device = sample_grad_list[0].device
    C = torch.tensor(C, device=device)
    sample_grad_clip_list = []

    with torch.no_grad():

        for b in range(batch_size):
            # concatenate grad from each parameter into a single vector
            g_b_cat = torch.cat([g[b].view(-1) for g in sample_grad_list])

            # Compute scaling factor for "clipping"
            scaling = 1 / torch.maximum(
                torch.tensor(1.0, device=device), g_b_cat.norm() / C
            )

            # scale the gradients of all the parameters using the scaling factor
            for i, g in enumerate(sample_grad_list):
                g_b_clip = g[b] * scaling
                if b == 0:
                    sample_grad_clip_list.append([g_b_clip])
                else:
                    sample_grad_clip_list[i].append(g_b_clip)

        # Compute the sum of sample-wise clipped gradients, add noise, compute mean
        grad_list_dp = []
        for g_batch_clip in sample_grad_clip_list:
            g_sum_clip = torch.sum(torch.stack(g_batch_clip, dim=0), dim=0)
            g_dp = (g_sum_clip / batch_size) + (
                torch.randn(g_sum_clip.shape, device=device) * sigma * C
            )
            grad_list_dp.append(g_dp)

    return grad_list_dp


def get_model_params(model):
    param_list = []
    for name, param in model.named_parameters():
        if "tok_emb" not in name:
            param_list.append(param)
    return param_list


def quantize_tensor_per_row_stochastic(x, bits=4, generator=None):
    if x.numel() == 0:
        return x
    orig_shape = tuple(x.shape)
    if x.ndim == 0:
        flat = x.reshape(1, 1)
    elif x.ndim == 1:
        flat = x.reshape(1, -1)
    else:
        flat = x.reshape(x.shape[0], -1)
    levels = (2 ** int(bits)) - 1
    radius = flat.detach().abs().amax(dim=1, keepdim=True)
    delta = 2.0 * radius / float(levels)
    nonzero = delta > 0
    delta_safe = torch.where(nonzero, delta, torch.ones_like(delta))
    noise = (torch.rand(flat.shape, dtype=flat.dtype, device=flat.device, generator=generator) - 0.5) * delta_safe
    q = torch.where(nonzero, flat.detach() + noise, torch.zeros_like(flat))
    half_width = delta / 2.0
    diff = flat.detach() - q
    inward = torch.minimum(half_width * 1.0e-3, torch.full_like(half_width, 1.0e-5))
    target_half_width = (half_width - inward).clamp_min(0.0)
    too_far = diff.abs() > target_half_width
    boundary = flat.detach() - diff.sign() * target_half_width
    q = torch.where(too_far, boundary, q)
    return q.reshape(orig_shape)


def quantize_grad_list_q4_stoch_perrow(grad_list, seed):
    out = []
    for idx, grad in enumerate(grad_list):
        gen = torch.Generator(device=grad.device)
        gen.manual_seed(int(seed) + 104729 * int(idx))
        out.append(quantize_tensor_per_row_stochastic(grad, bits=4, generator=gen))
    return out


def collect_grads(args):
    batch_size_list = [int(item) for item in str(args.batch_sizes).split(",") if item.strip()]

    # === Setup ===
    exp_path = f"{exp_path_base}/{args.ds}/{args.model}/grads/{args.defense}"
    logger = setup(exp_path=exp_path, log_file="collect_grad.log")
    logger.info(
        f"Collecting gradients with {args.model} model on {args.ds} with batch_sizes:"
        f" {batch_size_list} for {args.n_batch} batches"
    )
    device = get_device()

    # === Model ===
    dataparallel = False if (args.defense == "dp") else True
    model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        load_path=f"{exp_path_base}/{args.ds}/{args.model}/model.pt",
        dataparallel=dataparallel,
    )
    model_type = model.model_type
    model = model.to(device)
    if args.defense == "dp":
        if GradSampleModule is None:
            raise RuntimeError("opacus is required for --defense=dp")
        model = GradSampleModule(model)
    model.train()

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # === Collect grads ===
    for batch_size in batch_size_list:
        logger.info(f"\nCollecting grads with batch size: {batch_size}")
        _, dl_test = get_dataloaders(
            args.ds,
            batch_size=batch_size,
            shuffle_test=True,
            num_workers=int(args.num_workers),
        )

        data = {"x": [], "y": [], "z": [], "grad": [], "grad_clip": [], "grad_dp": []}

        for i, (x, y) in enumerate(dl_test):
            if i == args.n_batch:
                break
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

            if args.defense == "q4_stoch_perrow":
                grad = quantize_grad_list_q4_stoch_perrow(
                    grad,
                    seed=int(args.quant_seed) + 1009 * int(batch_size) + int(i),
                )

            grad = [g.cpu().numpy() for g in grad]
            data["grad"].append(grad)

        # Save data
        grads_pkl_file = get_grads_file(
            args.ds, args.model, args.defense, batch_size, args.C, args.sigma
        )
        os.makedirs(os.path.dirname(grads_pkl_file), exist_ok=True)
        if args.defense == "q4_stoch_perrow":
            data["quantization"] = {
                "bits": 4,
                "rounding": "stochastic",
                "scale_granularity": "per_row",
                "seed": int(args.quant_seed),
                "insertion_point": "collect_grads.grad_before_pickle",
            }
        write_pickle(data, grads_pkl_file)
        logger.info(f"Done!\nData saved in {grads_pkl_file}")


if __name__ == "__main__":
    logger = logging.getLogger()
    start = time.time()
    parser = argparse.ArgumentParser(description="Collect Gradients")
    parser.add_argument(
        "--ds",
        type=str,
        default="cifar10",
        help="dataset",
        choices=ds_dict.keys(),
    )
    parser.add_argument("--n_embd", type=int, default=100, help="Embedding Size")

    parser.add_argument(
        "--defense",
        type=str,
        default="nodef",
        help="Defense for gradient sharing",
        choices=["nodef", "dp", "q4_stoch_perrow"],
    )
    parser.add_argument("--quant_seed", type=int, default=20260908, help="Seed for q4_stoch_perrow")
    parser.add_argument(
        "--n_batch", type=int, default=10, help="number of batches to collect gradients"
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="8,16,32,64,128,256",
        help="comma-separated batch sizes to collect",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")

    parser.add_argument(
        "--model",
        type=str,
        default="fc2",
        help="model name",
        choices=model_class_dict.keys(),
    )
    parser.add_argument("--h_dim", type=int, default=256, help="hidden dim")

    parser.add_argument("--submitit", action="store_true", help="submit to queue")
    parser.add_argument("--n_cpu", type=int, default=40, help="cpus per task")
    parser.add_argument("--n_gpu", type=int, default=8, help="gpus per node")

    # DP
    parser.add_argument("--C", type=float, default=1.0, help="Clipping Norm")
    parser.add_argument("--sigma", type=float, default=0.0, help="sigmal for DP noise")

    args = parser.parse_args()

    # Submit it
    if args.submitit:
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=600,
            nodes=1,
            slurm_partition=slurm_partition,
            slurm_job_name=JOB_NAME,
            gpus_per_node=args.n_gpu,
            cpus_per_task=args.n_cpu,
            slurm_mem="400G",
        )
        print(JOB_NAME)
        job = executor.submit(collect_grads, args)
        print(f"Scheduled { job }.")
    else:
        collect_grads(args)

    end = time.time()
    runtime = end - start

    logger.info(f"\nRuntime: {runtime:.2f} s")
