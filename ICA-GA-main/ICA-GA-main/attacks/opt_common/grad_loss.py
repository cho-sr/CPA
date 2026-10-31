# --------------------------------------------------
# 梯度相似度损失计算模块
# --------------------------------------------------


from typing import List
import torch

from utils import logger


def cosine_distance(A: List, B: List) -> torch.Tensor:
    """两组梯度的余弦相似度"""
    A_flatten = torch.cat([x.view(-1) for x in A]).view(-1)
    B_flatten = torch.cat([y.view(-1) for y in B]).view(-1)
    costs = torch.dot(A_flatten, B_flatten) / (A_flatten.norm() * B_flatten.norm())
    costs = 1 - costs
    if torch.isnan(costs) or torch.isinf(costs):
        norm_A = A_flatten.norm().item()
        norm_B = B_flatten.norm().item()
        logger.warning(f"cosine_distance 计算异常！梯度范数：\n{norm_A}\n{norm_B}")
        for x in B:
            print(f"{x.view(-1).norm().item()}")
    return costs


def l2(A: List, B: List) -> torch.Tensor:
    """两组梯度的L2差异"""
    l2_loss = torch.tensor(0.0, device=A[0].device)
    for gx, gy in zip(A, B):
        l2_loss += ((gx - gy) ** 2).sum()
    return l2_loss


def normalize_l2(A: List, B: List) -> torch.Tensor:
    """模归一化L2差异"""
    A_flatten = torch.cat([x.view(-1) for x in A]).view(-1)
    B_flatten = torch.cat([y.view(-1) for y in B]).view(-1)
    norm_A = torch.norm(A_flatten, p=2)
    norm_B = torch.norm(B_flatten, p=2)
    eps = 1e-12  # 防止除零
    A_normed = A_flatten / (norm_A + eps)
    B_normed = B_flatten / (norm_B + eps)
    diff = (A_normed - B_normed).pow(2).sum()
    return diff
