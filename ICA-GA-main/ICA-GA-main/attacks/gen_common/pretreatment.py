# --------------------------------------------------
# 对特征向量的预处理
# --------------------------------------------------

import torch


def standardization(x: torch.Tensor):
    """对张量 [n, m] 中， n 组样本进行单独的标准化，使其均值为0标准差为1。"""
    means = x.mean(dim=1, keepdim=True)
    stds = x.std(dim=1, keepdim=True)
    standardized = (x - means) / stds
    return standardized


pretreatments = {
    "standardization": standardization,
}
