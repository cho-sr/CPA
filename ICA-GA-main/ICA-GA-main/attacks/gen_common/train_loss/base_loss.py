# --------------------------------------------------
# 训练生成器-损失项基类
# --------------------------------------------------

import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from ..generator_trainer import GeneratorTrainer


class LossInputDict(TypedDict):
    """输入损失项的字典

    Args:
        re_x: 重建样本
        real_x: 真实样本
        raw_feat: 真实特征
        generator_trainer: 训练器实例
    """

    re_x: torch.Tensor
    real_x: torch.Tensor
    raw_feat: torch.Tensor
    generator_trainer: "GeneratorTrainer"


class BaseLoss:
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, loss_input_dict: LossInputDict) -> torch.Tensor:
        pass
