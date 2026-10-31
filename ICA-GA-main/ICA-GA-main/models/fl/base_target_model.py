# --------------------------------------------------
# FL目标模型：基类
# --------------------------------------------------

import torch
import torch.nn as nn
from typing import Optional
from types import MethodType

from ..base_model import BaseModel
from utils import ParameterListDict


class BaseTargetModel(BaseModel):

    def __init__(
        self,
        input_width: int,
        input_height: int,
        output_length: int,
        input_channels: int = 3,
        *args,
        **kwargs,
    ):
        """初始化模型并进行参数验证

        Args:
            input_width (int): 实际输入图像宽度
            input_height (int): 实际输入图像高度
            output_length (int): 实际输出维度（分类数）
            input_channels (int): 实际输入图像通道数
        """
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.input_width = input_width
        self.input_height = input_height
        self.output_length = output_length

    def _get_image_output_size(self, model: nn.Module) -> list:
        """通过虚拟输入计算模型输出尺寸

        Args:
            model (nn.Module): 需要分析的模型

        Returns:
            list: 输出特征图尺寸（不含batch维度）
        """
        device = next(model.parameters()).device  # 获取模型所在设备
        input_tensor = torch.randn(
            1, self.input_channels, self.input_width, self.input_height, device=device
        )
        # 确保模型处于评估模式并禁用梯度
        original_training = model.training
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        model.train(original_training)  # 恢复原始模式

        return list(output.shape[1:])

    # =============== 辅助工具接口 ===============

    def get_parameter(
        self,
        device: Optional[torch.device] = None,
    ) -> ParameterListDict:
        """返回当前模型的参数的 ParameterListDict （副本数据）

        Args:
            device (torch.device): 非必须，将返回列表转移到指定计算设备。

        Returns:
            ParameterListDict: 当前模型的参数副本
        """
        params = []
        names = []
        for name, param in self.named_parameters():
            names.append(name)
            if device is not None:
                params.append(param.detach().to(device).clone())
            else:
                params.append(param.detach().clone())
        return ParameterListDict(params, names, self.__class__.__name__, device)

    def get_gradient(
        self,
        device: Optional[torch.device] = None,
    ) -> ParameterListDict:
        """返回当前模型的梯度副本的 ParameterListDict （副本数据）
        如果某个层不具有梯度，则返回的 ParameterListDict 中不含该层信息。

        Args:
            device (torch.device): 非必须，将返回列表转移到指定计算设备。

        Returns:
            ParameterListDict: 当前模型的梯度副本
        """
        grads = []
        names = []
        for name, param in self.named_parameters():
            if param.grad is None:  # 跳过没有梯度的层
                continue
            names.append(name)
            if device is not None:
                grads.append(param.grad.to(device).clone())
            else:
                grads.append(param.grad.clone())
        return ParameterListDict(
            grads,
            names,
            self.__class__.__name__,
        )

    def get_parameter_update(
        self,
        paras_0: ParameterListDict,
    ) -> ParameterListDict:
        """返回当前模型参数的更新量，可作为聚合梯度。
        如果某层没有梯度，那么返回值中不会包含该层，与 get_gradient 的行为一致。

        Args:
            paras_0: 初始梯度。将会计算 paras_0 - 当前参数

        Returns:
            ParameterListDict: 当前更新量
        """
        paras_1 = self.get_parameter()
        paras_u = paras_0 - paras_1
        gradient_names = self.get_gradient_names()
        return paras_u.get_subset(gradient_names)

    def get_gradient_names(self) -> list:
        """以列表形式，返回所有具有梯度的层的完整名称
        Returns:
            list: 参数名称列表，如 ['conv1.weight', 'conv1.bias']
        """
        names = []
        for name, param in self.named_parameters():
            if param.grad is None:  # 跳过没有梯度的层
                continue
            names.append(name)
        return names

    def get_parameter_names(self) -> list:
        """以列表形式，返回所有参数的完整名称
        Returns:
            list: 参数名称列表，如 ['conv1.weight', 'conv1.bias']
        """
        return [name for name, _ in self.named_parameters()]

    def get_parameter_index(self, name: str) -> int:
        """根据参数完整名称获取其在列表中的索引。可用于访问 get_parameter 等返回的列表
        Returns:
            int: 参数索引，找不到时返回-1
        """
        names = self.get_parameter_names()
        return names.index(name) if name in names else -1
