# --------------------------------------------------
# FL目标模型：VGG系列
# --------------------------------------------------

import torchvision
import torch.nn as nn

from .base_cnn_model import BaseCnnModel
from utils import logger


class Vgg16(BaseCnnModel):
    """VGG-16 架构（适配自定义全连接层）"""

    def __init__(
        self,
        input_width: int,
        input_height: int,
        output_length: int,
        input_channels: int = 3,
        is_pretrain: bool = False,
        is_avgpool: bool = True,
        is_bn: bool = True,
        fc_layers: list = [4096, "Dropout:0.5", 4096, "Dropout:0.5"],
        **kwargs,
    ):
        """初始化ResNet系列模型结构

        Args:
            is_pretrain (bool): 是否使用预训练参数
            is_avgpool (bool): 是否保留特征提取模块最后的平均池化层。
            - Tips: 对于 224 边长的图像， is_avgpool为True/False无影响，都相当于没有AvgPool。
            is_bn (bool): 是否保留模型中的batchnorm层
            input_channels (int): 实际输入通道数
            input_width (int): 实际输入图像宽度
            input_height (int): 实际输入图像高度
            output_length (int): 实际输出维度（分类数）
            fc_layers (tuple, list): 除了最后的分类层外，每个FC层的输出宽度。
                - 标准VGG: [4096, "Dropout:0.5", 4096, "Dropout:0.5"]

        Raises:
            ValueError: 参数不满足约束条件，或特征图尺寸无效时抛出
        """
        self.is_pretrain = is_pretrain
        self.is_avgpool = is_avgpool
        self.is_bn = is_bn
        super().__init__(
            input_width=input_width,
            input_height=input_height,
            output_length=output_length,
            input_channels=input_channels,
            fc_layers=fc_layers,
            **kwargs,
        )

    def _get_vgg(self) -> torchvision.models.VGG:
        """获取 vgg-16 模型，可在子类中重写"""
        pre_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        return torchvision.models.vgg16(
            weights=pre_weights if self.is_pretrain else None
        )

    def _remove_bn(self, module: nn.Module):
        """移除BN层"""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
            else:
                self._remove_bn(child)
        return module

    def _bulid_feature_extration(self) -> nn.Module:
        """获取VGG特征提取模块，移除原始全连接层。若is_avgpool=False则屏蔽最后的平均池化层"""
        vgg = self._get_vgg()
        # 冻结BN层
        if not self.is_bn:
            self._remove_bn(vgg)
            logger.debug(f"移除BN层：{self.__class__.__name__}")
        # 添加 AvgPool
        if self.is_avgpool:
            return nn.Sequential(vgg.features, vgg.avgpool)
        else:
            return vgg.features


class Vgg19(Vgg16):
    """Vgg19 架构（适配自定义全连接层）"""

    def _get_vgg(self) -> torchvision.models.VGG:
        pre_weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        return torchvision.models.vgg19(
            weights=pre_weights if self.is_pretrain else None
        )
