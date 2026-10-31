"""B=1/B=2 same80 FC-only T10 update: INT4 per-tensor vs per-channel scale.

User request 2026-10-12: with 1-2 images per local batch, compare reconstructions
against the original images when FC1's 4-bit scale is measured per tensor
versus per output channel (row). Same VGG16 start, frozen Conv, all six FC
trained with SGD lr=.001 for 10 full-batch steps, original upstream CPA (25k)
and Direct FIA (25k) settings from signed_baseline. Images are the first B of
the same80 batch00 (fixed by index, not by quality). Truth is used only to
build the sender update and for the offline evaluation after all arms finish.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import math
from copy import deepcopy
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from quantized_update_cpa import cell_utils
from . import upstream_naive
from .signed_baseline import CPA_CONFIG, FIA_CONFIG, SEED, ATTACK_LAYER, configure, verify_checkpoint, write_json, runtime
from .signed_baseline_observation import digest, CHECKPOINT_SHA

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT/"results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
OUT = ROOT/"results_private/int4_pc_vs_pt_B1_B2_T10_20261012"
MEAN, STD = (.485, .456, .406), (.229, .224, .225)


def per_channel(G):
    absmax = G.abs().amax(dim=1, keepdim=True)
    s = torch.where(absmax > 0, absmax/7, torch.ones_like(absmax))
    q = torch.round(G/s).clamp(-7, 7).to(torch.int8)
    return q, s


def