# --------------------------------------------------
# FL目标模型：卷积神经网络架构基类
# --------------------------------------------------

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Optional, Sequence

from .base_target_model import BaseTargetModel
from utils import logger


class BaseCnnModel(BaseTargetModel):
    """基于卷积神经网络的训练目标模型基类，结构包含：
    - self.feature_extration : 特征提取模块
    - self.fully_connected : 全连接层模块
    """

    def __init__(
        self,
        fc_layers: Optional[Sequence[int]] = None,
        *args,
        **kwargs,
    ):
        """初始化CNN模型结构

        Args:
            input_channels (int): 实际输入通道数
            input_width (int): 实际输入图像宽度
            input_height (int): 实际输入图像高度
            output_length (int): 实际输出维度（分类数）。如果为0，则不构造FC层
            fc_layers (tuple, list): 除了最后的分类层外，每个FC层的输出宽度，或Dropout。
                - 例： fc_layers=[] 表示只有1个FC层，输出宽度等于分类数 output_length
                - 例： fc_layers=[1024, 512] 表示有3个FC层，输出宽度分别是 1024, 512, output_length
                - 例： [4096, "Dropout:0.5", 4096, "Dropout:0.5"] 表示3个FC层，并插入2个Dropout
        """
        super().__init__(*args, **kwargs)
        # 构建特征提取模块
        self.feature_extration = self._bulid_feature_extration()
        # 构建全连接层模块
        if self.output_length > 0:
            self.fully_connected = self._build_fully_connected(fc_layers)
        else:
            self.fully_connected = nn.Sequential()
        # 缓存前向传播中的特征向量（展平后）
        self.feature_vector = None
        # 用于存储捕获的FC0梯度
        self._fc_0_dY = None
        # 用于持有钩子的句柄，方便移除
        self._hook_handle_fc_0 = None
        self.is_recording_fc0_dY = False  # 公开的状态标志

    # 【在子类中实现】
    @abstractmethod
    def _bulid_feature_extration(self) -> nn.Sequential:
        """抽象方法：构建特征提取模块实例（不含FC层）

        Returns:
            nn.Sequential: 特征提取主干网络实例
        """
        pass

    def _build_fully_connected(self, fc_layers: Optional[list] = None) -> nn.Sequential:
        """根据配置构建全连接层模块

        Args:
            fc_layers (list[int or str]):
                一个定义隐藏层结构的列表。
                - int: 指定一个 nn.Linear 层的输出维度。其后会自动跟随一个 ReLU 激活函数。
                - str: 格式为 "类型:参数" 的字符串。
                    - "Dropout:p": 添加一个 nn.Dropout(p=p) 层。p 为 0 到 1 之间的浮点数。
                - 最终的分类层由 self.output_length 自动确定，无需在列表中定义。
            None: 表示只含单个分类层。

        Returns:
            nn.Sequential: 全连接层序列

        Raises:
            ValueError: 当 fc_layers 包含非法维度值或无法解析的字符串时抛出。

        示例，VGG结构： [4096, "Dropout:0.5", 4096, "Dropout:0.5"]
        """
        if fc_layers is None:
            fc_layers = []
        fc = nn.Sequential()
        # 计算FC模块的输入维度，即特征提取模块的输出维度
        in_features = self._get_features_length()
        logger.debug(f"{self.__class__.__name__} in_features: {in_features}")

        # 使用独立的计数器为不同类型的层命名
        fc_count = 0
        dropout_count = 0

        for layer_spec in fc_layers:
            # Case 1: 整数 -> Linear + ReLU
            if isinstance(layer_spec, int):
                out_features = layer_spec
                if out_features <= 0:
                    raise ValueError("FC layer dimension must be a positive integer.")

                # 添加全连接层
                fc.add_module(f"fc_{fc_count}", nn.Linear(in_features, out_features))
                # 添加激活函数
                # Tips: 当需要使用 backward hook 时，其后的直接操作不能是 inplace 的。
                fc.add_module(f"relu_{fc_count}", nn.ReLU(inplace=False))

                # 为下一层更新输入维度
                in_features = out_features
                fc_count += 1

            # Case 2: 字符串 -> 特殊层 (例如 Dropout)
            elif isinstance(layer_spec, str):
                parts = layer_spec.lower().split(":")
                layer_type = parts[0]

                if layer_type == "dropout" and len(parts) == 2:
                    try:
                        p = float(parts[1])
                        if not (0.0 <= p < 1.0):
                            raise ValueError("Dropout probability must be in [0, 1).")
                        fc.add_module(f"dropout_{dropout_count}", nn.Dropout(p=p))
                        dropout_count += 1
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Invalid dropout value in '{layer_spec}'."
                        ) from e
                else:
                    raise ValueError(
                        f"Unsupported or malformed layer string: '{layer_spec}'"
                    )

            # Case 3: 不支持的类型
            else:
                raise TypeError(f"Unsupported type in fc_layers: {type(layer_spec)}")

        # 添加最后的分类层
        fc.add_module(f"fc_{fc_count}", nn.Linear(in_features, self.output_length))

        return fc

    def _get_features_length(self) -> int:
        """获取特征向量长度，即特征提取模块的输出向量的展平长度

        Returns:
            int: self.feature_extration 的输出向量的展平长度

        Raises:
            ValueError: 特征尺寸非法时抛出
        """
        # 计算特征向量尺寸，即特征提取模块的输出尺寸
        features_size = self._get_image_output_size(self.feature_extration)
        # 计算特征向量展平长度，作为首个FC层的输入宽度
        features_length = 1
        for dim in features_size:
            # 检查特征尺寸是否有效
            if dim <= 0:
                raise ValueError(
                    f"Invalid feature map size {features_size} with input channels {self.input_channels}, "
                    f"width {self.input_width}, height {self.input_height}. Please check input dimensions."
                )
            features_length *= dim
        return features_length

    # 前向传播
    def forward(self, x):
        """前向传播流程

        1. 特征提取
        2. 展平特征向量
        3. 全连接层
        """
        x = self.feature_extration(x)
        x = x.reshape(x.size(0), -1)
        self.feature_vector = x
        x = self.fully_connected(x)
        return x

    def get_feature_vector(self) -> torch.Tensor:
        """返回当前模型的特征向量的副本
        Returns:
            torch.Tensor: 当前模型的特征向量副本
        """
        if self.feature_vector is None:
            raise ValueError("feature_vector is None")
        return self.feature_vector.clone().detach()

    def _capture_fc_0_dY(
        self, module: nn.Module, grad_input: tuple, grad_output: tuple
    ):
        """钩子函数：捕获并存储首个FC层的 dY (输出梯度)"""
        # grad_output 是一个元组，对于线性层，它只包含一个张量
        self._fc_0_dY = grad_output[0]

    def enable_fc_0_dY_recording(self, enable: bool):
        """
        开启或关闭对 fc_0 层输出梯度的记录。

        Args:
            enable (bool): True 表示开启记录，False 表示关闭。
        """
        if enable:
            if self._hook_handle_fc_0 is not None:
                logger.warning("Recording is already enabled.")
                return

            # 检查 fc_0 是否存在
            if not hasattr(self.fully_connected, "fc_0"):
                raise AttributeError(
                    "Model does not have a layer named 'fc_0'. Cannot attach hook."
                )

            # 注册反向传播钩子并保存句柄
            target_layer = self.fully_connected.fc_0
            self._hook_handle_fc_0 = target_layer.register_full_backward_hook(
                self._capture_fc_0_dY
            )
            self.is_recording_fc0_dY = True
            logger.info("Enabled gradient recording for 'fc_0' output (dY).")
        else:
            if self._hook_handle_fc_0 is None:
                logger.warning("Recording is already disabled.")
                return

            # 使用句柄移除钩子
            self._hook_handle_fc_0.remove()
            self._hook_handle_fc_0 = None
            self._fc_0_dY = None  # 清理缓存的梯度
            self.is_recording_fc0_dY = False
            logger.info("Disabled gradient recording for 'fc_0'.")

    def get_fc_0_dY(self) -> Optional[torch.Tensor]:
        """
        获取 fc_0 层输出梯度 dY 的一个副本。

        只有在开启记录并执行过一次反向传播后才能获取到值。

        Returns:
            Optional[torch.Tensor]: 梯度张量的CPU副本，如果不可用则返回None。
        """
        if self._fc_0_dY is None:
            logger.warning(
                "Gradient for 'fc_0' is not available. "
                "Ensure recording is enabled and a backward pass has been performed."
            )
            return None

        return self._fc_0_dY.clone().cpu()

    def remove_bn(self, module: nn.Module):
        """移除BN层"""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
            else:
                self.remove_bn(child)
        return module

    def mask_modules(self, module_to_replace: str):
        """
        递归地将模型中指定类型的模块替换为nn.Identity()
        Args:
            module_to_replace (str): 要替换的模块的类名字符串，例如 "BatchNorm2d", "Dropout"。
        """
        num = 0

        def replace_modules_by_name(model: nn.Module, module_to_replace: str):
            nonlocal num
            # 遍历模型的所有直接子模块
            for name, child_module in model.named_children():
                # 检查子模块的类名是否与目标字符串匹配
                if child_module.__class__.__name__ == module_to_replace:
                    num += 1
                    # 如果匹配，使用 setattr 将该子模块替换为 nn.Identity
                    setattr(model, name, nn.Identity())
                else:
                    # 如果不匹配，递归进入该子模块继续查找
                    replace_modules_by_name(child_module, module_to_replace)

        replace_modules_by_name(self, module_to_replace)
        logger.debug(f"模型中屏蔽了 {num} 个 {module_to_replace} 模块")
