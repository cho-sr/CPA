# --------------------------------------------------
# 模块化生成器框架 Modular Generator
# --------------------------------------------------

from typing import List, Union
import torch.nn as nn

from ...base_model import BaseModel


class ModGen(BaseModel):

    # ================ 内部模块 ===============

    # 标准的卷积块
    class Conv(nn.Module):
        def __init__(self, channel_size):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )

        def forward(self, x):
            out = self.conv(x)
            return self.act(out)

    # 标准的残差卷积块
    class ResConv(Conv):

        def forward(self, x):
            out = self.conv(x) + x
            return self.act(out)

    # 带有 Squeeze-and-Excitation (SE) 模块的残差卷积块
    class ResConvSE(nn.Module):
        class SEBlock(nn.Module):
            def __init__(self, channel, reduction=16):
                super().__init__()
                # Squeeze 操作: 全局平均池化
                self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出 (B, C, 1, 1)

                # Excitation 操作: 两个全连接层 (用1x1卷积实现，保持数据的四维结构)
                self.fc = nn.Sequential(
                    # 第一个FC，降维
                    nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    # 第二个FC，升维
                    nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
                    nn.Sigmoid(),  # 输出通道权重，范围 (0, 1)
                )

            def forward(self, x):
                # Squeeze
                y = self.avg_pool(x)  # (B, C, 1, 1)
                # Excitation
                y = self.fc(y)  # (B, C, 1, 1)
                # Scale: 将权重y与输入x相乘
                # y.expand_as(x) 将 (B,C,1,1) 扩展为 (B,C,H,W)
                return x * y.expand_as(x)

        def __init__(self, channel_size, reduction=16):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv_block = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )
            self.se = self.SEBlock(channel_size, reduction)

        def forward(self, x):
            residual = x  # 保存原始输入用于残差连接
            out = self.conv_block(x)  # 通过卷积层
            out = self.se(out)  # 对卷积后的特征进行通道注意力调整
            out = out + residual  # 添加残差连接
            return self.act(out)  # 最后再激活

    # 使用 PixelShuffle 实现上采样的模块
    class PixelShuffleBlock(nn.Module):

        def __init__(self, in_channels, out_channels, scale_factor=2):
            super().__init__()
            # 卷积层输出 scale_factor^2 倍的通道数
            self.conv = nn.Conv2d(
                in_channels, out_channels * (scale_factor**2), kernel_size=3, padding=1
            )
            self.ps = nn.PixelShuffle(
                scale_factor
            )  # 将 (B, C*scale^2, H, W) 转换为 (B, C, H*scale, W*scale)

        def forward(self, x):
            return self.ps(self.conv(x))

    # ================ 整体模型结构 ===============

    def __init__(
        self,
        channels_schedule: List[int] = [2048, 1024, 512, 256, 128, 64],
        num_res_blocks_per_stage: Union[int, List[int]] = 3,
        res_block_type: str = "ResConv",
        upsample_type: str = "ConvTranspose",
        se_reduction: int = 16,
        final_out_channels: int = 3,
        initial_feature_map_size=(7, 7),
        *args,
        **kwargs,
    ):
        """
        可调节的生成器模型

        Args:
            channels_schedule (List[int]): 定义网络中每个阶段（上采样后）的通道数量。
                    第一个元素是第一个上采样/PixelShuffle块的输入通道数。
                    适配 ResNet-50 no pool: [2048, 1024, 512, 256, 128, 64]
                    适配 Vgg : [512, 512, 256, 128, 64, 32]
            num_res_blocks_per_stage (Union[int, List[int]]): 每个上采样阶段后的残差块数量。
                                                           如果为 int，则所有阶段使用相同数量的残差块。
                                                           如果为 list，其长度应等于 (len(channels_schedule) - 1)。
            res_block_type (str): 残差块的类型。可选值: 'Conv', 'ResConv', 'ResConvSE'。
            upsample_type (str): 上采样层的类型。可选值: 'ConvTranspose', 'PixelShuffle'。
            se_reduction (int): 当 res_block_type 为 'ResConvSE' 时，SE模块中通道缩减的比例。
            final_out_channels (int): 模型最终输出的通道数 (例如，RGB图像为3)。
            initial_feature_map_size (tuple of int): 输入到 self.conv 网络之前的特征图的空间维度 (H, W)。
                                                     用于在 forward 方法中正确 reshape 输入张量。
            *args, **kwargs: 传递给 BaseModel 构造函数的额外参数。
        """
        super(ModGen, self).__init__(*args, **kwargs)

        self.channels_schedule = channels_schedule
        self.initial_feature_map_size = initial_feature_map_size
        # 用于 forward 方法中的 view/reshape 操作
        self.initial_channels = channels_schedule[0]

        layers = []  # 用于存储网络层
        num_stages = len(channels_schedule) - 1  # 上采样阶段的数量

        # 校验并格式化 num_res_blocks_per_stage
        if isinstance(num_res_blocks_per_stage, int):
            num_res_blocks_list = [num_res_blocks_per_stage] * num_stages
        elif (
            isinstance(num_res_blocks_per_stage, list)
            and len(num_res_blocks_per_stage) == num_stages
        ):
            num_res_blocks_list = num_res_blocks_per_stage
        else:
            raise ValueError(
                f"num_res_blocks_per_stage ({num_res_blocks_per_stage}, {type(num_res_blocks_per_stage)}) 必须是一个整数，"
                f"或者是一个长度为 num_stages ({num_stages}) 的整数列表。"
            )

        # 构建网络的主体部分
        for i in range(num_stages):
            # 对于每一组基础块：
            in_channels = channels_schedule[i]
            out_channels = channels_schedule[i + 1]

            # 1. 上采样层
            if upsample_type == "ConvTranspose":
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                )
            elif upsample_type == "PixelShuffle":
                layers.append(
                    self.PixelShuffleBlock(in_channels, out_channels, scale_factor=2)
                )
            else:
                raise ValueError(f"不支持的上采样类型: {upsample_type}")

            layers.append(nn.LeakyReLU(inplace=True))  # 每个上采样后的激活函数

            # 2. 残差块序列
            current_res_block_count = num_res_blocks_list[i]
            for _ in range(current_res_block_count):
                if res_block_type == "Conv":
                    layers.append(self.Conv(out_channels))
                elif res_block_type == "ResConv":
                    layers.append(self.ResConv(out_channels))
                elif res_block_type == "ResConvSE":
                    layers.append(self.ResConvSE(out_channels, reduction=se_reduction))
                else:
                    raise ValueError(f"不支持的残差块类型: {res_block_type}")

        # 最终输出层
        # 使用 1x1 卷积将通道数调整为所需的 final_out_channels
        layers.append(
            nn.Conv2d(
                channels_schedule[-1], final_out_channels, kernel_size=1, padding=0
            )
        )
        # 输出激活函数，将值映射到 (0, 1) 范围
        layers.append(nn.Sigmoid())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
                              可以是 (batch_size, initial_channels * H * W) 的二维张量，
                              或者 (batch_size, initial_channels, H, W) 的四维张量。
                              其中 H, W 由 self.initial_feature_map_size 定义。
        """
        # 根据 initial_channels 和 initial_feature_map_size 调整输入张量形状
        # 预期的扁平化特征数量
        expected_flat_features = (
            self.initial_channels
            * self.initial_feature_map_size[0]
            * self.initial_feature_map_size[1]
        )

        if x.ndim == 2 and x.shape[1] == expected_flat_features:
            # 输入是 (batch_size, C*H*W)，需要 reshape
            out = x.view(
                -1,
                self.initial_channels,
                self.initial_feature_map_size[0],
                self.initial_feature_map_size[1],
            )
        elif (
            x.ndim == 4
            and x.shape[1] == self.initial_channels
            and x.shape[2] == self.initial_feature_map_size[0]
            and x.shape[3] == self.initial_feature_map_size[1]
        ):
            # 输入已经是 (batch_size, C, H, W) 的正确形状
            out = x
        else:
            raise ValueError(
                f"输入张量形状 {x.shape} 不兼容。 "
                f"期望一个扁平化的张量，形状为 (batch_size, {expected_flat_features})，"
                f"或者一个四维张量，形状为 (batch_size, {self.initial_channels}, "
                f"{self.initial_feature_map_size[0]}, {self.initial_feature_map_size[1]})."
            )

        out = self.conv(out)
        return out
