# --------------------------------------------------
# FL目标模型：ResNet系列
# --------------------------------------------------

import torchvision
import torch.nn as nn

from .base_cnn_model import BaseCnnModel
from utils import logger


class ResNet18(BaseCnnModel):
    """ResNet18 架构（适配自定义全连接层）"""

    def __init__(
        self,
        is_pretrain: bool = False,
        is_avgpool: bool = True,
        is_bn: bool = True,
        is_pretrain_fc=False,
        *args,
        **kwargs,
    ):
        """初始化ResNet系列模型结构

        Args:
            is_pretrain (bool): 特征提取器是否使用预训练参数
            is_avgpool (bool): 是否保留特征提取模块最后的平均池化层
            is_bn (bool): 是否保留模型中的batchnorm层
            is_pretrain_fc (bool): 分类器是否使用预训练参数（使fc_layers无效，output_length=1000）
            input_channels (int): 实际输入通道数
            input_width (int): 实际输入图像宽度
            input_height (int): 实际输入图像高度
            output_length (int): 实际输出维度（分类数）
            fc_layers (tuple, list): 除了最后的分类层外，每个FC层的输出宽度。
                - 例： fc_layers=[] 表示只有1个FC层，输出宽度等于分类数 output_length
                - 例： fc_layers=[1024, 512] 表示有3个FC层，输出宽度分别是 1024, 512, output_length

        Raises:
            ValueError: 参数不满足约束条件，或特征图尺寸无效时抛出
        """
        self.is_pretrain = is_pretrain
        self.is_avgpool = is_avgpool
        self.is_bn = is_bn
        self.is_pretrain_fc = is_pretrain_fc
        super().__init__(*args, **kwargs)
        # 保留预训练FC层
        if self.is_pretrain_fc:
            self.fully_connected = self._pretrain_fc

    def _get_resnet(self) -> torchvision.models.ResNet:
        """获取resnet18模型，可在子类中重写"""
        pre_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        return torchvision.models.resnet18(
            weights=pre_weights if self.is_pretrain else None
        )

    def _bulid_feature_extration(self) -> nn.Sequential:
        """获取ResNet特征提取模块，移除原始全连接层。若is_avgpool=False则屏蔽最后的平均池化层"""
        resnet = self._get_resnet()
        # 移除平均池化层
        if not self.is_avgpool:
            resnet.avgpool = nn.Identity()
        # 冻结BN层
        if not self.is_bn:
            self.remove_bn(resnet)
            logger.debug(f"移除BN层：{self.__class__.__name__}")
        # 保留预训练FC层
        if self.is_pretrain_fc:
            self._pretrain_fc = resnet.fc
        return nn.Sequential(*list(resnet.children())[:-1])


class ResNet50(ResNet18):
    """ResNet50 架构（适配自定义全连接层）"""

    def _get_resnet(self) -> nn.Module:
        """获取resnet50模型"""
        pre_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        return torchvision.models.resnet50(
            weights=pre_weights if self.is_pretrain else None
        )


"""
# 调用示例
resnet18 = ResNet18(
    input_channels=3,  # 3通道
    input_width=224,
    input_height=224,
    output_length=100,  # 100 分类
    fc_layers=[2048],  # 增加一个2048的FC隐藏层，最终生成2个FC层： [2048, 100]
)
print(resnet18.get_parameter_names())
"""
