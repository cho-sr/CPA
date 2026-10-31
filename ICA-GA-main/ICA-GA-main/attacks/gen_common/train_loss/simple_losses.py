# --------------------------------------------------
# 常用的简单损失项
# --------------------------------------------------


import torch
import torch.nn as nn

from .base_loss import BaseLoss, LossInputDict
from ...opt_common.regularization import total_variation
from utils.pytorch_ssim import ssim as pytorch_ssim


class MseLoss(BaseLoss):
    """像素级均方误差损失 (MSE Loss)"""

    def __init__(self, *args, **kwargs):
        self._mse_loss_fn = nn.MSELoss()

    def __call__(self, loss_input_dict: LossInputDict):
        re_x = loss_input_dict["re_x"]
        real_x = loss_input_dict["real_x"]
        return self._mse_loss_fn(re_x, real_x)


class FeatLoss(BaseLoss):
    """特征向量相似度损失 (Feature Loss)"""

    def __init__(self, *args, **kwargs):
        self._mse_loss_fn = nn.MSELoss()

    def __call__(self, loss_input_dict: LossInputDict):
        encoder = loss_input_dict["generator_trainer"].encoder
        re_x = loss_input_dict["re_x"]
        re_feat = encoder(re_x)
        raw_feat = loss_input_dict["raw_feat"]
        return self._mse_loss_fn(re_feat, raw_feat)


class TvLoss(BaseLoss):
    """总变分损失 (Total Variation Loss)"""

    def __call__(self, loss_input_dict: LossInputDict):
        re_x = loss_input_dict["re_x"]
        return total_variation(re_x)


class SsimLoss(BaseLoss):
    """结构相似性损失 (Total Variation Loss)"""

    def __call__(self, loss_input_dict: LossInputDict):
        re_x = loss_input_dict["re_x"]
        real_x = loss_input_dict["real_x"]
        return 1 - pytorch_ssim(re_x, real_x)
