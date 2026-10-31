# --------------------------------------------------
# 正则化模块
# --------------------------------------------------

import torch


def total_variation(A: torch.Tensor):
    """TV正则项"""
    dx = torch.mean(torch.abs(A[:, :, :, :-1] - A[:, :, :, 1:]))
    dy = torch.mean(torch.abs(A[:, :, :-1, :] - A[:, :, 1:, :]))
    return dx + dy
